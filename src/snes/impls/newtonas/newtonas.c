
#include <../src/snes/impls/newtonas/newtonasimpl.h> /*I "petscsnes.h" I*/
#include <petsc-private/kspimpl.h>
#include <petsc-private/matimpl.h>
#include <petsc-private/dmimpl.h>
#include <petsc-private/vecimpl.h>

#define SNES_NEWTONAS_WORK_N 3
/*
   Usese of snes->work:
   snes->work[0:3] - SNESNEWTONASMeritFunction()
   snes->work[0]   - SNESNEWTONASComputeSearchDirectionSaddle_Private()
 */

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASMonitorDefault"
/*@C
   SNESNEWTONASMonitorDefault - Monitors progress of the SNES active set NEWTONAS solver.

   Collective on SNES

   Input Parameters:
+  snes   - the SNES context
.  its    - iteration number
.  fnorm  - 2-norm of residual (ignored)
-  viewer - optional viewer to send the monitor output to

   Notes:
   This routine prints the value of the merit function used by the active set Newton solver at each iteration.
   The merit function value is stored in the SNESNEWTONAS object.

   Level: intermediate

.keywords: SNES, nonlinear, default, monitor, norm, merit function

.seealso: SNESMonitorSet(), SNESMonitorDefault()
@*/
PetscErrorCode  SNESNEWTONASMonitorDefault(SNES snes,PetscInt its,PetscReal fnorm,void *view)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = view ? (PetscViewer) view : PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));
  SNES_NEWTONAS  *newtas = (SNES_NEWTONAS*)snes->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%3D SNESNEWTONAS merit function %14.12e \n",its,(double)newtas->merit);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASInitialActiveSet_Private"
static PetscErrorCode SNESNEWTONASInitialActiveSet_Private(SNES snes,IS *active)
{
  PetscErrorCode     ierr;
  SNES_NEWTONAS      *newtas = (SNES_NEWTONAS*)snes->data;
  PetscInt           glo,ghi;
  PetscInt           i;
  Vec                g,gl,gu;
  const PetscScalar  *la,*ua,*ga,*lam;
  PetscInt           *indices,counter=0;
  Vec                l = newtas->vec_lambda;
  MPI_Comm           comm;

  PetscFunctionBegin;
  /* Assume that f(x) and g(x) have already been computed */
  /* A = \{i : (g_i(x) <= g_i^l+epsilon & \lambda_i > 0) | (g_i(x) >= g_i^u-epsilon & \lambda_i < 0)\} -- strongly active set. */
  *active = NULL;
  ierr = SNESGetConstraintFunction(snes,&g,&gl,&gu,NULL,NULL);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(l,&glo,&ghi);CHKERRQ(ierr);

  ierr = VecGetArrayRead(gl,&la);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gu,&ua);CHKERRQ(ierr);
  ierr = VecGetArrayRead(g,&ga);CHKERRQ(ierr);
  ierr = VecGetArrayRead(l,&lam);CHKERRQ(ierr);
  ierr = PetscCalloc1(ghi-glo,&indices);CHKERRQ(ierr);
  for (i=0;i<ghi-glo;i++) {
    if (((PetscRealPart(ga[i]) <= PetscRealPart(la[i])) &&
        (PetscRealPart(lam[i]) > 0)) ||
        ((PetscRealPart(ga[i]) >= PetscRealPart(ua[i])) &&
         (PetscRealPart(lam[i] < 0)))) {
      indices[i] = i+glo; counter++;
    }
  }
  ierr = VecRestoreArrayRead(gl,&la);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(gu,&ua);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(g,&ga);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(l,&lam);CHKERRQ(ierr);
  if (counter > 0) {
    ierr = PetscObjectGetComm((PetscObject)g,&comm);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,counter,indices,PETSC_OWN_POINTER,active);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASModifyActiveSet_Private"
static PetscErrorCode SNESNEWTONASModifyActiveSet_Private(SNES snes,IS active,IS *new_active,PetscReal *tbar)
{
  /* Returns a modified IS based on the distance to constraint bounds, or NULL if no modification is necessary. */
  SNES_NEWTONAS     *newtas  = (SNES_NEWTONAS*)snes->data;
  Vec               dx=snes->vec_sol_update,dl=newtas->vec_lambda_update;
  Vec               gl=snes->vec_constrl, gu=snes->vec_constru;
  Vec               bx = newtas->work_Bdx;
  PetscInt          i,lo,hi;
  const PetscScalar *gl_v,*gu_v,*g_v,*l_v,*dl_v,*bx_v;
  PetscReal         tlimit,tilimit,bdxi,umgi,lmgi,ldli;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /*
     Compute tbar -- the upper bound on the search step size.
     tbar_i = largest t such that for i in inactive set
                                  g_i(x) + t*(B*dx)_i <= gu_i AND
                                  g_i(x) + t*(B*dx)_i >= gl_i AND

                                  for i in active set:
                                  if at upper bound:
                                       l_i + t*dl_i < 0
                                  if at lower bound:
                                       l_i + t*dl_i > 0
               this is equivalent to:
     tbar_i = largest t such that:  t >= 0
                                    if (Bdx_i > 0):
                                        t >= (l_i - g_i)/(Bdx_i) (always neg)
                                        t <= (u_i - g_i)/(Bdx_i) (always pos)
                                    if (Bdx_i < 0):
                                        t <= (l_i - g_i)/(Bdx_i) (always pos)
                                        t >= (u_i - g_i)/(Bdx_i) (always neg)
                                    if (g_i == l_i): (lower bound)
                                        (make sure l>0)
                                        if (dl_i > 0)
                                           t > -l_i / dl_i (always neg)
                                        else
                                           t < -l_i / dl_i (always pos)
                                    if (g_i == u_i): (upper bound)
                                        (make sure l<0)
                                        if (dl_i > 0)
                                           t < -l_i / dl_i (always positive)
                                        else
                                           t > -l_i / dl_i (always negative)

           Removing redundancies:
        inactive  If Bdx_i > 0:               t <= (u_i - g_i)/Bdx_i
        inactive  If Bdx_i < 0:               t <= (l_i - g_i)/Bdx_i
         active   if g_i==l_i && dl_i>0
                        OR                    t < -l_i / dl_i
         active   if g_i==u_i && dl_i<0





     tbar = MPI_Allreduce(tbar_i).
     If tbar == 0, BARF.

     g(x) = snes->vec_constr
     gu   = snes->vec_constru
     glu  = snes->vec_constrl
     Use newtas->work_Bdx to store the result of B*dx calls.
     B = snes->jacobian_constr.

     x = snes->vec_sol,
     f(x) = snes->vec_func
     (dx,dl) = (snes->vec_sol_update,newtas->vec_lambda_update)
  */
  *new_active = NULL;
  *tbar = 0.0;

  ierr = SNESGetConstraintFunction(snes,NULL,&gl,&gu,NULL,NULL);CHKERRQ(ierr);
  ierr = MatMult(snes->jacobian_constrt,dx,bx);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(dl,&lo,&hi);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gl,&gl_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gu,&gu_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->vec_constr,&g_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(newtas->vec_lambda,&l_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(dl,&dl_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bx,&bx_v);CHKERRQ(ierr);
  tlimit = PETSC_INFINITY;

  for (i=0;i<hi-lo;i++) {
    umgi = PetscRealPart(gu_v[i] - g_v[i]);
    lmgi = PetscRealPart(gl_v[i] - g_v[i]);

    if (umgi < PETSC_SQRT_MACHINE_EPSILON ||
        -lmgi < PETSC_SQRT_MACHINE_EPSILON ) {
      ldli = -l_v[i] / dl_v[i];
      tilimit = PetscRealPart(ldli);
    } else {
      bdxi = PetscRealPart(bx_v[i]);
      if (bdxi > 0) {
        tilimit = umgi/bdxi;
      } else if (bdxi < 0) {
        tilimit = lmgi/bdxi;
      } else {
        tilimit=0;
      }
    }
    tlimit = PetscMin(tilimit,tlimit);
  }

  ierr = VecRestoreArrayRead(gl,&gl_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(gu,&gu_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->vec_constr,&g_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(newtas->vec_lambda,&l_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(dl,&dl_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bx,&bx_v);CHKERRQ(ierr);

  ierr = MPI_Allreduce(&tlimit,tbar,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)snes->vec_constr));CHKERRQ(ierr);

  if (*tbar <= 0) {
    /*TODO - handle degeneracy */
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"tbar (%g) <= 0 in SNESNEWTONASModifyActiveSet_Private\n",*tbar);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASScatter"
/*
   SNESNEWTONASScatter - Takes the augmented solution (e.g., from the line search) and separates
                   it into the solution and lagrange multiplier componenents
*/
static PetscErrorCode SNESNEWTONASScatter(SNES snes,Vec X_aug,Vec X,Vec Lambda)
{

  SNES_NEWTONAS     *newtas  = (SNES_NEWTONAS*)snes->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(newtas->aug_to_x,X_aug,X,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(newtas->aug_to_lambda,X_aug,Lambda,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(newtas->aug_to_x,X_aug,X,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(newtas->aug_to_lambda,X_aug,Lambda,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASGather"
/*
   SNESNEWTONASGather - Combines the solution vector and lagrange multipliers
                    into one vector
*/
static PetscErrorCode SNESNEWTONASGather(SNES snes,Vec X_aug,Vec X,Vec Lambda)
{
  SNES_NEWTONAS     *newtas  = (SNES_NEWTONAS*)snes->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(newtas->aug_to_x,X,X_aug,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(newtas->aug_to_lambda,Lambda,X_aug,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(newtas->aug_to_x,X,X_aug,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(newtas->aug_to_lambda,Lambda,X_aug,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASMeritFunction"
/* QUESTION: static of PETSC_INTERNAL? */
static PetscErrorCode SNESNEWTONASMeritFunction(SNES snes, Vec X, PetscReal *f)
{
  SNES_NEWTONAS      *newtas = (SNES_NEWTONAS*)snes->data;
  Vec                workx   = snes->work[0];
  Vec                workf   = snes->work[1];
  Vec                workBtl = snes->work[2];
  Vec                workl   = newtas->work_lambda;
  PetscErrorCode     ierr;

  PetscFunctionBegin;

  ierr = SNESNEWTONASScatter(snes,X,workx,workl);CHKERRQ(ierr);
  /* TODO: check vec_func_init_set here? */
  ierr = SNESComputeFunction(snes,workx,workf);CHKERRQ(ierr);

  /* ||f - l*B||_2, if B == NULL, treat it as zero. */
  if (snes->jacobian_constrt || snes->jacobian_constr) {
    if (snes->jacobian_constrt) {
      ierr = MatMult(snes->jacobian_constrt,workl,workBtl);CHKERRQ(ierr);
    } else if (snes->jacobian_constr) {
      ierr = MatMultTranspose(snes->jacobian_constr,workl,workBtl);CHKERRQ(ierr);
    }
    ierr = VecAXPY(workf,-1.0,workBtl);CHKERRQ(ierr);
  }
  ierr = VecNorm(workf,NORM_2,f);CHKERRQ(ierr);
  /* QUESTION: Jason, why are we returning the square of the norm? */
  *f *= *f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASComputeSearchDirectionSaddle_Private"
static PetscErrorCode SNESNEWTONASComputeSearchDirectionSaddle_Private(SNES snes,Vec x,Vec l,IS active,Vec dx,Vec dl,PetscInt *lits)
{
  PetscErrorCode     ierr;
  SNES_NEWTONAS      *newtas = (SNES_NEWTONAS*)(snes->data);
  Mat                Bt;
  IS                 row[3],col[3];

  PetscFunctionBegin;
  if (!newtas->jacobian_aug) {
    ierr = MatCreate(PetscObjectComm((PetscObject)snes),&newtas->jacobian_aug);CHKERRQ(ierr); /* FIXME: destroy jacobian_aug in the d'tor. */
    ierr = MatSetType(newtas->jacobian_aug,MATNEST);CHKERRQ(ierr);
    if (!snes->jacobian_constrt) {
      ierr = MatTranspose(snes->jacobian_constr,MAT_INITIAL_MATRIX,&Bt);CHKERRQ(ierr); /* FIXME: reuse the MatNest block, whenever possible. */
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASComputeSearchDirectionPrimal_Private"
static PetscErrorCode SNESNEWTONASComputeSearchDirectionPrimal_Private(SNES snes,Vec x,Vec l,IS active,Vec dx,Vec dl,PetscInt *lits)
{
  /* Observe that only a subvector of dl defined by the active set is nonzero. */
  /* PetscErrorCode     ierr; */
  /* SNES_NEWTONAS      *newtas = (SNES_NEWTONAS*)(snes->data); */
  /* Vec                tildedl; */ /* \tilde \delta \lambda */
  /* Vec                q = snes->work[3]; */  /* q = -(f - B^T*l) */

  PetscFunctionBegin;
#if 0
  /* The following code needs to be cleaned up so that it at least compiles. */
    /* Compute the primal and constraint Jacobians to form the reduced linear system below. */
    ierr = SNESComputeJacobian(snes,x,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    /*
       TODO: eventually we want something like SNESComputeConstraintJacobian(),
       which will handle the MF case of the constraint Jacobian, comparison to the
       explicitly-computed operator and debugging.
    */
    ierr = dmsnes->ops->constraintjacobian(snes,x,snes->jacobian_constr,snes->jacobian_constrt,dmsnes->constraintjacobianctx);CHKERRQ(ierr);


  /*
    This function computes the linear update in the reduced subspace (defined by the active index set)
     of the augmented system (defined by the state x and Lagrange multipliers l).  The linear update
     is computed by eliminating the constraints using the 'active basis' computed by SNESNEWTONASActiveConstraintBasis().
  */
  ierr = VecZeroEntries(dl);CHKERRQ(ierr);


    /*
      Conceptually, we are solving the augmented system.
                           |A         \tilde B^T| |    dx     |   |-(f-B^T*l)|
                           |                    | |           | = |          |
			   |\tilde B           0| | \tilde dl |   |    0     |
      In reality we eliminate \tilde dlambda using Bb and Bbt and solve for dx1 with the Schur complement.
      dx0 and tdlambda are recovered later and assembled into dx and dl:
                           |   A00        A01     \tilde B0^T| |   dx0     |   |-(f0-B0^T*l)|
                           |   A10        A11     \tilde B1^T| |   dx1     | = |-(f1-B1^T*l)|
			   |\tilde B0   \tilde B1      0     | | \tilde dl |   |      0     |,

      where \tilde B0 is Bb -- the submatrix composed of the basis columns.  For clarity, reorder:

                           |\tilde B0      0        \tilde B1| | \tilde dl |   |      0     |
                           |   A00     \tilde B0^T     A01   | |    dx0    | = |-(f0-B0^T*l)|
			   |   A10     \tilde B1^T     A11   | |    dx1    |   |-(f1+B1^T*l)|

      Now eliminate the upper-left 2x2 block submatrix, which can be factored as follows:
          |\tilde B0       0     |       |\tilde B0       0     | |    I       0     | |   I      0      |
      K = |                      |   =   |                      | |                  | |                 |
          |   A00     \tilde B0^T|	 |    0           I     | |   A00      I     | |   0  \tilde B0^T|

      Hence, the iverse is:

                  |\tilde B0^{-1}                                0       |
       K^{-1} =   |                                                      |
                  |-\tilde B0^{-T}A00\tilde B0^{-1}        \tilde B0^{-T}|

      Finally, the Schur complement resulting from this elimination is:
                                          |\tilde B1|
      S = A11 - [A10 \tilde B1^T] K^{-1}  |         |
                                          |   A01   |

        = A11  + \tilde B1^T\tilde B0{-T} A00 \tilde B0^{-1} \tilde B1- A10 \tilde B0^{-1} \tilde B1 - \tilde B1^T \tilde B0^{-T} A01

    */
    /* FIXME: We need to outfit KSP with a suitable DM built out of the primal DM attached by the user
       to SNES and out of the constraint DM, which isn't currently being attached to SNES.
       If we were solving the saddle-point problem directly, without elimination, a saddle-point DM
       would be needed.  For the primal problem (as here) we need a DM that resembles the primal DM
       that we already have.  If we were solving the dual problem, the KSP DM would need to be modeled
       on the constraint DM.

    */
  /*
     TODO: form J as a MatNest (possibly followed by a MatConvert()) of [A \tilde B^T; \tilde B 0].
  */
  ierr = KSPSetOperators(snes->ksp,J,J);CHKERRQ(ierr);
  /* TODO: configure snes->ksp using PCFIELDSPLIT to implement the solve with S above. */
  ierr = KSPSetUp(snes->ksp);CHKERRQ(ierr);
  /* TODO: embed tildedl into dl and then splice dx and dl into y */
  ierr = KSPSolve(snes->ksp,q,y);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
  if (kspreason < 0) {
    if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
      ierr         = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
      snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
      break;
    }
  }
#endif
  PetscFunctionReturn(0);
}

/* SNES NEWTONAS ALGORITHM SUBROUTINE STUBS BEGIN */


#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NEWTONAS"
PetscErrorCode SNESSolve_NEWTONAS(SNES snes)
{
  PetscErrorCode     ierr;
  Vec                x,dx,f,l,dl,x_aug,dx_aug,f_aug;
  SNES_NEWTONAS      *newtas = (SNES_NEWTONAS*)snes->data;
  DM                  dm;
  DMSNES              dmsnes;
  PetscInt            i,lits;
  PetscReal           fnorm,xnorm,dxnorm,hnorm;
  PetscReal           merit;
  PetscBool           lssucceed;
  PetscBool           domainerror;
  PetscReal           tbar;
  SNESLineSearch      linesearch=snes->linesearch;
  IS                  active,new_active;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  x      = snes->vec_sol;               /* solution vector */
  f      = snes->vec_func;              /* residual vector */
  dx     = snes->vec_sol_update;        /* newton step */
  l      = newtas->vec_lambda;          /* \lambda */
  dl     = newtas->vec_lambda_update;   /* \delta \lambda */
  x_aug  = newtas->vec_soln_aug;        /* augmented solution vector */
  f_aug  = newtas->vec_func_aug;        /* augmented residual vector */
  dx_aug = newtas->vec_soln_update_aug; /* augmented update vector */


  ierr            = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter      = 0;
  snes->norm      = 0.0;
  snes->merit     = 0.0;
  ierr            = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  /*
     TODO: should this be a SNESNEWTONAS-specific DMSNES? Should this stuff be attached to dm_aug, perhaps?
     How is thie DMSNES context to be shared among the multiple DMs potentially associated with
     a constrained problem (dm, constraint_dm, saddle_dm)?
  */
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDMSNES(dm,&dmsnes);CHKERRQ(ierr);

  /* TODO: Provide SNESComputeProjectOntoConstraints()?  Or is it a purely private function?
     It could be called manually after extracting the callback using SNESGetComputeOntoConstraints().
  */
  if (dmsnes->ops->projectontoconstraints) {
    ierr = (*dmsnes->ops->projectontoconstraints)(snes,x,x,dmsnes->projectontoconstraintsctx);CHKERRQ(ierr);
  } /* No 'else' clause since there is really no default way of projecting onto constraints that I know of. */


  /*
     TODO: nonlinear LEFT PC application would go here. Project afterwards as well?
  */

  /*
     TODO: the following is a premature optimization, because it doesn't interact well with SNESNETONASMeritFunction(),
     which recomputes f anyway.  How can this optimization be restored?
     One way to do it is to pass both x and f to SNESNEWTONASMeritFunction() and fall back onto snes->vec_soln, snes->vec_func,
     if they are NULL.  This is a slippery slope, however, since then we can start passing the other data members, like B,Bt, etc.

     QUESTION: the 'else' below is a puzzling clause: if snes->vec_func_init_set is true, set it to false?  Why?  For later iterations or subsolves?
  */
  /*
  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes,x,f);CHKERRQ(ierr);
    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
  } else snes->vec_func_init_set = PETSC_FALSE;
  */

  ierr          = SNESNEWTONASGather(snes,x_aug,x,l);CHKERRQ(ierr);
  ierr          = SNESNEWTONASMeritFunction(snes,x_aug,&merit);CHKERRQ(ierr);
  ierr          = VecNorm(f,NORM_2,&fnorm);CHKERRQ(ierr);
  ierr          = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm    = fnorm;
  newtas->merit = merit;
  ierr          = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  /* TODO: should we log merit instead?  In addition? Is SNESLogConvergenceHistory() sensitive to the meaning of that argument? */
  ierr          = SNESLogConvergenceHistory(snes,merit,0);CHKERRQ(ierr);
  ierr          = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* test convergence */
  /* HACK: for now pass merit in place of fnorm. */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,merit,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<snes->max_its; ++i) {
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    /*
       TODO: nonlinear RIGHT PC application would go here.
    */


    new_active = NULL;
    ierr = SNESNEWTONASInitialActiveSet_Private(snes,&active);CHKERRQ(ierr);
    do {
      if (new_active) { /* active set has been updated */
	    ierr = ISDestroy(&active);CHKERRQ(ierr);
	    active = new_active; new_active = NULL;
      }
      if (newtas->type == SNESNEWTONAS_SADDLE) {
	/* Need to shove x,l into vec_sol, dx, dl into vec_sol_update. */
	ierr = SNESNEWTONASComputeSearchDirectionSaddle_Private(snes,x,l,active,dx,dl,&lits);CHKERRQ(ierr);
      }
      else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"SNESNEWTONAS solver type not yet supported: %s",newtas->type);
      ierr = SNESNEWTONASModifyActiveSet_Private(snes,active,&new_active,&tbar);CHKERRQ(ierr);
    } while (new_active);

    /* Compute a (scaled) negative update in the line search routine:
         x_aug <- x_aug - alpha*dx_aug
       and evaluate f = function(x) (depends on the line search).
    */
    /* FIXME: what is hnorm? */
    hnorm = fnorm;

    /* FIXME: Jason,Todd, what should we be passing in for f_aug?  Anything? The below stuff is almost certainly incorrect. */
    ierr = SNESNEWTONASGather(snes,x_aug,x,l);CHKERRQ(ierr);
    ierr = VecScale(x_aug,tbar);CHKERRQ(ierr);
    ierr = SNESNEWTONASGather(snes,f_aug,f,l);CHKERRQ(ierr);
    ierr = SNESNEWTONASGather(snes,dx_aug,dx,dl);CHKERRQ(ierr);
    /* TODO: what can linesearch do with fnorm?  Should the merit function be passed in and out instead?
     Should we use whatever norm is stashed on the corresponding vector, whenever possible?
     The merit function, however, isn't cached anyplace convenient, since it's the norm of a
     transient vector.  We pass in NULL for now.
    */
    ierr = SNESLineSearchApply(linesearch,x_aug,f_aug,NULL,dx_aug);CHKERRQ(ierr);
    ierr = SNESNEWTONASScatter(snes,x_aug,x,l);CHKERRQ(ierr);
    ierr = SNESLineSearchGetSuccess(linesearch, &lssucceed);CHKERRQ(ierr);
    ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &dxnorm);CHKERRQ(ierr);
    /* FIXME: need to extract the merit function so that we can monitor its convergence. */
    /* TODO:  roll merit into SNESLineSearchGetNorms(), even though merit might not be a norm? */
    /* ierr = SNESLineSearchGetMerit(linesearch,&merit);CHKERRQ(ierr); */
    ierr  = PetscInfo5(snes,"merit=%18.16e, fnorm=%18.16e, hnorm=%18.16e, dxnorm=%18.16e, lssucceed=%d\n",(double)merit,(double)hnorm,(double)fnorm,(double)dxnorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;

    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      /* QUESTION: is doing this outside of the convergence test a hack? */
      if (snes->stol*xnorm > dxnorm) {
        snes->reason = SNES_CONVERGED_SNORM_RELATIVE;
        PetscFunctionReturn(0);
      }
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin=PETSC_FALSE;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        /*
	   TODO: replace this with a NEWTONAS-specific check?
	   ierr         = SNESNEWTONLSCheckLocalMin_Private(snes,snes->jacobian,f,w,fnorm,&ismin);CHKERRQ(ierr);
	*/
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    if (dmsnes->ops->projectontoconstraints) {
      ierr = (*dmsnes->ops->projectontoconstraints)(snes,x,x,dmsnes->projectontoconstraintsctx);CHKERRQ(ierr);
      ierr = SNESNEWTONASGather(snes,x_aug,x,l);CHKERRQ(ierr);
      ierr = SNESNEWTONASMeritFunction(snes,x_aug,&merit);CHKERRQ(ierr);

    } /* No 'else' clause since there is really no default way of projecting onto constraints that I know of. */

    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    snes->merit= merit;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    /* TODO: should we log merit instead?  In addition? Is SNESLogConvergenceHistory() sensitive to the meaning of that argument? */
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,fnorm);CHKERRQ(ierr);
    /* Test for convergence */
    /*
      HACK:  Currently we substitute the merit function for the norm of the residual. We need to incorporate the merit function
             into the convergence test in a BACKWARD-compatible way.
    */
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,dxnorm,newtas->merit,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == snes->max_its) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}



#undef  __FUNCT__
#define __FUNCT__ "SNESNEWTONASSetType"
/*@C
  SNESNEWTONASSetType - set the type of the solver handling the saddle-point problem
  underlying the active-set Newton linesearch method solving the nonlinear constrained
  (variational inequality) problem.

  Logically collective on SNES.

  Input Parameters:
+ snes  - the SNES context
- type  - SNESNEWTONAS solver type

  Level: intermediate

  Options Database Keys: -snes_newtonas_type primal|dual|saddle

.keywords: SNES constraints, saddle-point

.seealso: SNESSetType(), SNESNEWTONASGetType()
@*/
PetscErrorCode SNESNEWTONASSetType(SNES snes,SNESNEWTONASType type)
{
  /*PetscErrorCode    ierr; */
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  newtas->type = type;
  snes->ops->solve = SNESSolve_NEWTONAS;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SNESNEWTONASGetType"
/*@C
  SNESNEWTONASGetType - retrieve the type of the solver handling the saddle-point problem
  underlying the active-set Newton linesearch method solving the nonlinear constrained
  (variational inequality) problem.

  Not collective

  Input parameter:
. snes  - the SNES context

  Output parameter:
. type  - SNESNEWTONAS solver type

  Level: intermediate

 .keywords: SNES constraints, saddle-point

.seealso: SNESGetType(), SNESNEWTONASSetType()
@*/
PetscErrorCode SNESNEWTONASGetType(SNES snes,SNESNEWTONASType *type)
{
  /* PetscErrorCode    ierr; */
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  if (type) *type = newtas->type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESActiveConstraints_Default"
PETSC_INTERN PetscErrorCode SNESActiveConstraints_Default(SNES snes,Vec x,IS *active,IS *basis,void *ctx)
{
  /* PetscErrorCode    ierr; */
  /* SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data; */

  PetscFunctionBegin;
  /* TODO: Apply QR or SVD to a redundant B? */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NEWTONAS"
PETSC_INTERN PetscErrorCode SNESSetFromOptions_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;
  PetscViewer       monviewer;
  char              monfilename[PETSC_MAX_PATH_LEN+1];
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNESNEWTONAS solver options");CHKERRQ(ierr);
  newtas->type = SNESNEWTONAS_PRIMAL;
  ierr = PetscOptionsEnum("-snes_newtonas_type","Type of linear solver to use for the saddle-point problem","SNESNEWTONASSetType",SNESNEWTONASTypes,(PetscEnum)newtas->type,(PetscEnum*)&newtas->type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-snes_newtonas_monitor","Monitor active set merit function","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)snes),monfilename,&monviewer);CHKERRQ(ierr);
    ierr = SNESMonitorSet(snes,SNESNEWTONASMonitorDefault,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NEWTONAS"
PETSC_INTERN PetscErrorCode SNESSetUp_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  PetscInt          xlo,xhi,llo,lhi;
  IS                is_x,is_l,is_aug_x,is_aug_l;
  SNESLineSearch    linesearch;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  if (!newtas->type) {
    ierr = SNESNEWTONASSetType(snes,SNESNEWTONAS_SADDLE);CHKERRQ(ierr);
  }


  /* Set up constraint and bound vectors, if not provided. */
  if (!snes->vec_constr) {
    if (snes->dm_constr) {
      ierr = DMCreateGlobalVector(snes->dm_constr,&snes->vec_constr);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No constraint vector.  Call SNESSetConstraintFunction() or SNESSetConstraintDM().");
  }
  /* FIXME: Set up bound vectors, if not provided (-Inf/Inf). */


  ierr = VecDuplicate(snes->vec_constr,&newtas->vec_lambda);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_constr,&newtas->vec_lambda_update);CHKERRQ(ierr);

  /*
    TODO: Do we need to go through the public API to set private data structures?
    The rationale given in the docs is that this gives control to plugins, but wouldn't
    they have to go into the SNES data structures to make use of the work vecs?
    A related question: why not let the impl allocated and clean up its own work vecs?
  */
  ierr = SNESSetWorkVecs(snes,SNES_NEWTONAS_WORK_N);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_constr,&newtas->work_Bdx);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_constr,&newtas->work_lambda);CHKERRQ(ierr);

  ierr = VecCreate(((PetscObject)snes)->comm,&newtas->vec_soln_aug);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(snes->vec_sol,&xlo,&xhi);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(newtas->vec_lambda,&llo,&lhi);CHKERRQ(ierr);

  ierr = VecSetSizes(newtas->vec_soln_aug,lhi-llo + xhi-xlo, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(newtas->vec_soln_aug,((PetscObject)(snes->vec_sol))->type_name);CHKERRQ(ierr);
  ierr = VecSetFromOptions(newtas->vec_soln_aug);CHKERRQ(ierr);
  ierr = VecDuplicate(newtas->vec_soln_aug,&newtas->vec_func_aug);CHKERRQ(ierr);
  ierr = VecDuplicate(newtas->vec_soln_aug,&newtas->vec_soln_update_aug);CHKERRQ(ierr);

  ierr = VecDestroy(&newtas->work_Bdx);CHKERRQ(ierr);
  ierr = VecDestroy(&newtas->work_lambda);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_constr,&newtas->work_Bdx);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_constr,&newtas->work_lambda);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,xhi-xlo,xlo,1,&is_x);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,xhi-xlo,xlo+llo,1,&is_aug_x);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,lhi-llo,llo,1,&is_l);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,lhi-llo,xhi+llo,1,&is_aug_l);CHKERRQ(ierr);

  ierr = VecScatterCreate(newtas->vec_soln_aug,is_aug_x,snes->vec_sol,is_x,&newtas->aug_to_x);CHKERRQ(ierr);
  ierr = VecScatterCreate(newtas->vec_soln_aug,is_aug_l,newtas->vec_lambda,is_l,&newtas->aug_to_lambda);CHKERRQ(ierr);
  ierr = ISDestroy(&is_x);CHKERRQ(ierr);
  ierr = ISDestroy(&is_l);CHKERRQ(ierr);
  ierr = ISDestroy(&is_aug_x);CHKERRQ(ierr);
  ierr = ISDestroy(&is_aug_l);CHKERRQ(ierr);

  ierr = SNESGetLineSearch(snes,&linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBTSD);CHKERRQ(ierr);
  ierr = SNESLineSearchSetMerit(linesearch,SNESNEWTONASMeritFunction);CHKERRQ(ierr);

  /* FIXME: set up jacobian_constr, jacobian_constr, and jacobian_aug, if not provided. */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NEWTONAS"
PETSC_INTERN PetscErrorCode SNESDestroy_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  ierr = VecDestroy(&newtas->vec_lambda);CHKERRQ(ierr);
  ierr = VecDestroy(&newtas->vec_lambda_update);CHKERRQ(ierr);

  ierr = VecDestroy(&newtas->vec_soln_aug);CHKERRQ(ierr);
  ierr = VecDestroy(&newtas->vec_func_aug);CHKERRQ(ierr);
  ierr = VecDestroy(&newtas->vec_soln_update_aug);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&newtas->aug_to_x);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&newtas->aug_to_lambda);CHKERRQ(ierr);

  ierr = VecDestroy(&newtas->work_Bdx);CHKERRQ(ierr);
  ierr = VecDestroy(&newtas->work_lambda);CHKERRQ(ierr);

  /* FIXME: clean up jacobians. */

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESNEWTONAS - Augmented-space active-set linesearch-based Newton-like solver
      for nonlinear problems with constraints (variational inequalities or mixed
      complementarity problems).

   Options Database:
.   -snes_newtonas_type primal|dual|saddle


   Level: beginner

   References:
   - T. S. Munson, and S. Benson. Flexible Complementarity Solvers for Large-Scale
     Applications, Optimization Methods and Software, 21 (2006).

.seealso:  SNES, SNESCreate(), SNESSetType(), SNESLineSearchSet(),
           SNESSetFunction(), SNESSetJacobian(), SNESSetConstraintFunction(), SNESSetConstraintJacobian(),
           SNESSetActiveConstraints(), SNESSetProjectOntoConstraints(), SNESSetDistanceToConstraints(),
           SNESNEWTONASSetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NEWTONAS"
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONAS;
  snes->ops->destroy        = SNESDestroy_NEWTONAS;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONAS;

  ierr                = PetscNewLog(snes,&newtas);CHKERRQ(ierr);
  snes->data          = (void*)newtas;

  PetscFunctionReturn(0);
}


const char *const SNESNEWTONASTypes[] = {"PRIMAL","SADDLE","DUAL","SNESNEWTONASType","SNESNEWTONAS_",0};
/*MC
    SNESNEWTONASActiveConstraintBasis - callback function identifying a basis for the active constraints
    linearized at the current state vector x of the constrained nonlinear problem (variational inequality)
    solved by SNES

     Synopsis:
     #include <petscsnes.h>
     SNESNEWTONASActiveConstraintBasis(SNES snes,Vec x,Vec f,Vec g,Vec B,IS active,IS *basis,Mat Bb_pre,Bbt_pre,void *ctx);

     Input Parameters:
+     snes   - the SNES context
.     x      - state at which to evaluate activities
.     f      - function at x
.     g      - constraints at x
.     B      - constraint Jacobian at x
.     active - set of active constraint indices
-     ctx    - optional user-defined function context, passed in with SNESSetActiveConstraintBasis()

     Output Parameters:
+     basis   - indices of basis vectors spanning the active linearized constraint range
.     Bb_pre  - (NULL, if not available) preconditioning matrix for Bb
-     Bbt_pre - (NULL, if not available) preconditioning matrix for Bbt

     Notes:
     The active linearized constraint range is the range of the columns of B. Output parameter 'basis'
     comprises the indices of B's columns that are a basis for the active constraint linearized constraint
     range. Bb is the square matrix with these column indices, so the columns of Bb are the basis of the
     linearized constraint range. Since in primal elimination methods inverses (or solves with) of both Bb
     and Bbt, the transpose of Bb, are needed, the user can provide matrices to build preconditioners for
     both Bb and Bbt.


   Level: intermediate

.seealso:   SNESNEWTONASSetAcitveConstraintBasis(), SNESNEWTONASSetActiveConstraints(), SNESSetConstraintFunction(), SNESSetConstraintJacobian(), SNESConstraintFunction, SNESConstraintJacobian

 M*/

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASSetActiveConstraintBasis"
/*@C
   SNESNEWTONASSetActiveConstraintBasis -   sets the callback identifying a basis for linearized active constraints.

   Logically Collective on SNES

   Input Parameters:
+  snes    - the SNES context
.  Bb_pre  - (NULL, if not provided) matrix to store the preconditioner for the basis for the active constraints
.  Bbt_pre - (NULL, if not provided) transposed basis matrix for the inactive constraints
.  f       - function identifying the active constraint basis at the current state x
-  ctx     - optional (if not NULL) user-defined context for private data for the
          identification of active constraints function.

   Level: intermediate

.keywords: SNES, nonlinear, set, active, constraint, function

.seealso: SNESNEWTONASGetActiveConstraintBasiss(), SNESSetConstraintFunction(), SNESNEWTONASActiveConstraintBasis
@*/
PetscErrorCode  SNESNEWTONASSetActiveConstraintBasis(SNES snes,Mat Bb_pre, Mat Bbt_pre,PetscErrorCode (*f)(SNES,Vec,Vec,Vec,Mat,IS,IS*,Mat,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  SNES_NEWTONAS  *newtas = (SNES_NEWTONAS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  /* FIXME: Check this is a SNESNEWTONAS */
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESNEWTONASSetActiveConstraintBasis(dm,f,ctx);CHKERRQ(ierr);
  if (Bb_pre) {
    ierr = PetscObjectReference((PetscObject)Bb_pre);CHKERRQ(ierr);
    ierr = MatDestroy(&newtas->Bb_pre);CHKERRQ(ierr);
    newtas->Bb_pre = Bb_pre;
  }
  if (Bbt_pre) {
    ierr = PetscObjectReference((PetscObject)Bbt_pre);CHKERRQ(ierr);
    ierr = MatDestroy(&newtas->Bbt_pre);CHKERRQ(ierr);
    newtas->Bbt_pre = Bbt_pre;
  }
  PetscFunctionReturn(0);
}
