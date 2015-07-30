
#include <../src/snes/impls/newtonaugas/newtonaugasimpl.h> /*I "petscsnes.h" I*/
#include <petsc-private/snesimpl.h>
#include <petsc-private/kspimpl.h>
#include <petsc-private/matimpl.h>
#include <petsc-private/dmimpl.h>
#include <petsc-private/vecimpl.h>


#define SNES_NEWTONAUGAS_WORK_N 3
/*
   Uses of snes->work:
   snes->work[0:2] - SNESNEWTONAUGASComputeAugResidual(): x,f,B^T*l
   snes->work[0]   - SNESNEWTONAUGASComputeSearchDirection_Private(): g=B^T*l
 */

#define SNES_NEWTONAUGAS_WORK_CONSTR_N 2
/*
   Uses of snes->work_constr:
   snes->work_constr[0] - SNESNEWTONAUGASComputeAugResidual(): workl
   snes->work_constr[1] - SNESNEWTONAUGASComputeAugResidual(): workg
 */

#define SNES_NEWTONAUGAS_WORK_AUG_N 3
/*
   Uses of snes->work_aug:
   snes->work_aug[0:2] -  SNESSolve_NEWTONAUGAS: x_aug,dx_aug,f_aug
   snes->work_aug[0]   -  SNESNEWTONAUGASMeritFunction(): workaug
   snes->work_aug[0:1] -  SNESNEWTONAUGASComputeSearchDirection_Private: dx_aug,f_aug.
 */

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONAUGASMonitorDefault"
/*@C
   SNESNEWTONAUGASMonitorDefault - Monitors progress of the SNES active set NEWTONAUGAS solver.

   Collective on SNES

   Input Parameters:
+  snes   - the SNES context
.  its    - iteration number
.  fnorm  - 2-norm of residual
-  viewer - optional viewer to send the monitor output to

   Notes:
   This simply computes the norm of the augmented residual.


   Level: intermediate

.keywords: SNES, nonlinear, default, monitor, norm, augmented residual

.seealso: SNESMonitorSet(), SNESMonitorDefault()
@*/
PetscErrorCode  SNESNEWTONAUGASMonitorDefault(SNES snes,PetscInt its,PetscReal fnorm,void *view)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = view ? (PetscViewer) view : PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%3D SNESNEWTONAUGAS augmented residual norm %14.12e \n",its,(double)fnorm);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONAUGASInitialActiveSet_Private"
static PetscErrorCode SNESNEWTONAUGASInitialActiveSet_Private(SNES snes,IS *active)
{
  PetscErrorCode     ierr;
  SNES_NEWTONAUGAS      *newtaugas = (SNES_NEWTONAUGAS*)snes->data;
  PetscInt           glo,ghi;
  PetscInt           i;
  Vec                g,gl,gu;
  const PetscScalar  *la,*ua,*ga;
  PetscScalar        *lam;
  PetscInt           *indices,counter=0;
  Vec                l = newtaugas->vec_sol_lambda;
  MPI_Comm           comm;

  PetscFunctionBegin;
  /* Assume that f(x) and g(x) have already been computed */
  /* A = \{i : (g_i(x) <= g_i^l+epsilon & \lambda_i > 0) | (g_i(x) >= g_i^u-epsilon & \lambda_i < 0)\} -- strongly active set. */

  ierr = ISDestroy(active);CHKERRQ(ierr);

  ierr = SNESConstraintGetFunction(snes,&g,&gl,&gu,NULL,NULL);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(l,&glo,&ghi);CHKERRQ(ierr);

  ierr = VecGetArrayRead(gl,&la);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gu,&ua);CHKERRQ(ierr);
  ierr = VecGetArrayRead(g,&ga);CHKERRQ(ierr);
  ierr = VecGetArray(l,&lam);CHKERRQ(ierr);
  ierr = PetscCalloc1(ghi-glo,&indices);CHKERRQ(ierr);
  if (snes->iter==0) {
    for (i=0;i<ghi-glo;i++) {
      if (PetscRealPart(la[i]-ga[i]) >= 0) {
        lam[i] = 1.0;
      }
      if (PetscRealPart(ga[i]-ua[i]) >= 0) {
        lam[i] = -1.0;
      }
    }
  } else {
    for (i=0;i<ghi-glo;i++) {
      if (((PetscRealPart(ga[i]) <= PetscRealPart(la[i])) &&
         ((PetscRealPart(lam[i]) > 0) || snes->iter==0)) ||
        ((PetscRealPart(ga[i]) >= PetscRealPart(ua[i])) &&
         (PetscRealPart(lam[i] < 0) || snes->iter==0))) {
        indices[i] = i+glo; counter++;
      }
    }
  }
  ierr = VecRestoreArrayRead(gl,&la);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(gu,&ua);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(g,&ga);CHKERRQ(ierr);
  ierr = VecRestoreArray(l,&lam);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)g,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,counter,indices,PETSC_OWN_POINTER,active);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONAUGASModifyActiveSet_Private"
static PetscErrorCode SNESNEWTONAUGASModifyActiveSet_Private(SNES snes,IS active,IS *new_active,PetscReal *tbar)
{
  /* Returns a modified IS based on the distance to constraint bounds, or NULL if no modification is necessary. */
  SNES_NEWTONAUGAS     *newtaugas  = (SNES_NEWTONAUGAS*)snes->data;
  /*  TODO NOW: why is dx=0? */
  Vec               dx=snes->vec_sol_update,dl=newtaugas->vec_sol_lambda_update;
  Vec               gl=snes->vec_constrl, gu=snes->vec_constru;
  Vec               bx = snes->work_constr[0];
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
     Use newtaugas->work_Bdx to store the result of B*dx calls.
     B = snes->jacobian_constr.

     x = snes->vec_sol,
     f(x) = snes->vec_func
     (dx,dl) = (snes->vec_sol_update,newtaugas->vec_lambda_update)
  */
  *new_active = NULL;
  *tbar = 0.0;

  ierr = SNESConstraintGetFunction(snes,NULL,&gl,&gu,NULL,NULL);CHKERRQ(ierr);
  if (snes->jacobian_constrt) {
    ierr = MatMult(snes->jacobian_constrt,dx,bx);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(snes->jacobian_constr,dx,bx);CHKERRQ(ierr);
  }
  ierr = VecGetOwnershipRange(dl,&lo,&hi);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gl,&gl_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gu,&gu_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->vec_constr,&g_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(newtaugas->vec_sol_lambda,&l_v);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(newtaugas->vec_sol_lambda,&l_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(dl,&dl_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bx,&bx_v);CHKERRQ(ierr);

  ierr = MPI_Allreduce(&tlimit,tbar,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)snes->vec_constr));CHKERRQ(ierr);

  if (*tbar <= 0) {
    /*TODO - handle degeneracy */
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"tbar (%g) <= 0 in SNESNEWTONAUGASModifyActiveSet_Private\n",*tbar);
  }

  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONAUGASComputeAugResidual"
static PetscErrorCode SNESNEWTONAUGASComputeAugResidual(SNES snes, Vec X, Vec R)
{
  Vec                workx   = snes->work[0];
  Vec                workl   = snes->work_constr[0];
  Vec                workf   = snes->work[1];
  Vec                workBtl = snes->work[2];
  PetscErrorCode     ierr;

  PetscFunctionBegin;

  ierr = SNESConstraintAugScatter(snes,X,workx,workl);CHKERRQ(ierr);
  /* TODO: check vec_func_init_set here? */
  ierr = SNESConstraintComputeFunctions(snes,workx,R,workf,NULL);CHKERRQ(ierr);
  ierr = SNESConstraintComputeJacobians(snes,workx,MAT_REUSE_MATRIX,snes->jacobian_aug,NULL,snes->jacobian_aug_shape,NULL,NULL,snes->jacobian_constr,snes->jacobian_constrt);CHKERRQ(ierr);
  /* ||f - l*B||_2, if B == NULL, treat it as zero. */
  if (snes->jacobian_constrt || snes->jacobian_constr) {
    if (snes->jacobian_constrt) {
      ierr = MatMult(snes->jacobian_constrt,workl,workBtl);CHKERRQ(ierr);
    } else if (snes->jacobian_constr) {
      ierr = MatMultTranspose(snes->jacobian_constr,workl,workBtl);CHKERRQ(ierr);
    }
    ierr = VecAXPY(workf,-1.0,workBtl);CHKERRQ(ierr);
  }
  ierr = VecSet(R,0.0);CHKERRQ(ierr);
  ierr = SNESConstraintAugGather(snes,R,workf,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONAUGASMeritFunction"
static PetscErrorCode SNESNEWTONAUGASMeritFunction(SNES snes, Vec X, PetscReal *f)
{
  Vec                workaug = snes->work_aug[0];
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = SNESNEWTONAUGASComputeAugResidual(snes,X,workaug);CHKERRQ(ierr);
  ierr = VecNorm(workaug,NORM_2,f);CHKERRQ(ierr);
  /* QUESTION: Jason, why are we returning the square of the norm? */
  *f *= *f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONAUGASMakeAugIS_Private"
/* Map the subsets of primal and dual index spaces into the aug space using the corresponding embeddings. */
static PetscErrorCode SNESNEWTONAUGASMakeAugIS_Private(SNES snes,IS activep,IS activec,IS *activep_aug,IS *activec_aug)
{
  /*
  const PetscInt *arrayc,*array_constr_emb;
  PetscInt       sizep,sizec,clo,chi;
  PetscErrorCode ierr;
   */
  PetscFunctionBegin;
  /* CONTINUE: implement */
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONAUGASComputeSearchDirection_Private"
static PetscErrorCode SNESNEWTONAUGASComputeSearchDirection_Private(SNES snes,IS activec,PetscInt *lits)
{
  PetscErrorCode     ierr;
  SNES_NEWTONAUGAS   *newtaugas = (SNES_NEWTONAUGAS*)(snes->data);
  Vec                l=newtaugas->vec_sol_lambda,dx_aug=snes->work_aug[0],f_aug=snes->work_aug[1],h=snes->work[1],f=snes->vec_func;
  KSPConvergedReason kspreason;
  PetscErrorCode     (*func)(SNES,Vec,Vec,void*),(*afunc)(SNES,Vec,Vec,void*);
  Mat                Jr_aug, Jr_aug_pre;

  PetscFunctionBegin;
  ierr = SNESGetFunction(snes,NULL,&func,NULL);CHKERRQ(ierr);
  ierr = SNESConstraintGetAugFunction(snes,NULL,NULL,NULL,&afunc,NULL);CHKERRQ(ierr);

  /*
    Conceptually, we are solving the reduced augmented system.
    |A         \tilde B^T| | dx |   |-(f-\tilde B^T*l)|
    |                    | |    | = |                 |
    |\tilde B           0| | dl |   |    0            |
  */
  /* Create the reduced reduced-augmented Jacobian and its preconditioning matrix. */
  /* FIXME: reuse datastructures whenever possible, if the active sets haven't changed. */
  /* CONTINUE: embed active sets -- primal (box) and dual (constraints) -- into the aug index spaces; use XXXMakeAugIS() and call these activep_aug and ativec_aug. */
  if (afunc) {
    /* CONTINUE:
       build Jr_aug_is -- the IS to extract the active reduced Jr_aug (both row and col).
       Use ISConcatenate() on the complement of the activep_aug (or the primal embedding is_func_aug, if the primal active set is NULL), and activec_aug.
       Construct the Jr_aug.
     */
  }
  if (func) {
    /* CONTINUE: build Jr_aug as a MatNest out of A, active B and active Bt */
  }
  /* CONTINUE: Attach activep_aug, activec_aug and A,B,Bt submatrices to Jr_aug, Jr_aug_pre so that solvers like PCSaddle can take advantage of it. */
  ierr = KSPSetOperators(newtaugas->kspr_aug,Jr_aug,Jr_aug_pre);CHKERRQ(ierr);
  ierr = KSPSetUp(newtaugas->kspr_aug);CHKERRQ(ierr);
  if (snes->jacobian_constrt) {
    ierr = MatMult(snes->jacobian_constrt,l,h);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(snes->jacobian_constr,l,h);CHKERRQ(ierr);
  }
  ierr = VecAXPY(h,1.0,f);CHKERRQ(ierr);
  ierr = VecScale(h,-1.0);CHKERRQ(ierr);
  ierr = SNESConstraintAugGather(snes,f_aug,h,NULL);CHKERRQ(ierr);

  ierr = KSPSolve(newtaugas->kspr_aug,f_aug,dx_aug);CHKERRQ(ierr);
  ierr = SNESConstraintAugScatter(snes,dx_aug,snes->vec_sol_update,newtaugas->vec_sol_lambda_update);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(newtaugas->kspr_aug,&kspreason);CHKERRQ(ierr);
  if (kspreason < 0) {
    if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
      ierr         = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
      snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
    }
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NEWTONAUGAS"
PetscErrorCode SNESSolve_NEWTONAUGAS(SNES snes)
{
  PetscErrorCode      ierr;
  SNES_NEWTONAUGAS    *newtaugas = (SNES_NEWTONAUGAS*)snes->data;
  Vec                 x,dx,f,l,dl,x_aug,dx_aug,f_aug;
  DM                  dm;
  DMSNES              dmsnes;
  PetscInt            i,lits,nactive,idx;
  const PetscInt      *iactive;
  PetscReal           fnorm,xnorm,dxnorm,hnorm;
  PetscReal           merit,*larray,tbar;
  PetscBool           lssucceed,domainerror,preview;
  SNESLineSearch      linesearch=snes->linesearch;
  IS                  active=NULL,new_active;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  x      = snes->vec_sol;                   /* solution vector */
  f      = snes->vec_func;                  /* residual vector */
  dx     = snes->vec_sol_update;            /* newton step */
  l      = newtaugas->vec_sol_lambda;          /* \lambda */
  dl     = newtaugas->vec_sol_lambda_update;   /* \delta \lambda */
  f_aug  = snes->vec_func_aug;              /* augmented residual vector */
  x_aug  = snes->work_aug[0];               /* augmented solution vector */
  dx_aug = snes->work_aug[1];               /* augmented update vector */

  /*
   TODO!: get rid of merit, use the 2-norm of the augmented residual (a workaug).
   */

  ierr            = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter      = 0;
  snes->norm      = 0.0;
  snes->merit     = 0.0;
  ierr            = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);


  /*
     TODO: should this be a SNESNEWTONAUGAS-specific DMSNES? Should this stuff be attached to dm_aug, perhaps?
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
     One way to do it is to pass both x and f to SNESNEWTONAUGASMeritFunction() and fall back onto snes->vec_soln, snes->vec_func,
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
  ierr          = PetscOptionsHasName(((PetscObject)snes)->prefix,"-snes_newtonaugas_preview_sol",&preview);CHKERRQ(ierr);
  if (preview) {
    ierr          = PetscPrintf(PetscObjectComm((PetscObject)x),"Initial solution vector:\n");CHKERRQ(ierr);
    ierr          = VecViewFromOptions(x,((PetscObject)snes)->prefix,"-snes_newtonaugas_preview_sol");CHKERRQ(ierr);
  }

  ierr          = SNESConstraintComputeFunctions(snes,x,snes->vec_func,snes->vec_constr,snes->vec_func_aug);CHKERRQ(ierr);

  if (snes->vec_func) {
    ierr          = PetscOptionsHasName(((PetscObject)snes)->prefix,"-snes_newtonaugas_preview_func",&preview);CHKERRQ(ierr);
    if (preview) {
      ierr          = PetscPrintf(PetscObjectComm((PetscObject)snes->vec_func),"Initial function vector:\n");CHKERRQ(ierr);
      ierr          = VecViewFromOptions(snes->vec_func,((PetscObject)snes)->prefix,"-snes_newtonaugas_preview_func");CHKERRQ(ierr);
    }
  }
  if (snes->vec_constr) {
    ierr          = PetscOptionsHasName(((PetscObject)snes)->prefix,"-snes_newtonaugas_preview_constr",&preview);CHKERRQ(ierr);
    if (preview) {
      ierr          = PetscPrintf(PetscObjectComm((PetscObject)snes->vec_constr),"Initial constraint vector:\n");CHKERRQ(ierr);
      ierr          = VecViewFromOptions(snes->vec_constr,((PetscObject)snes)->prefix,"-snes_newtonaugas_preview_constr");CHKERRQ(ierr);
    }
  }
  if (snes->vec_func_aug) {
    ierr          = PetscOptionsHasName(((PetscObject)snes)->prefix,"-snes_newtonaugas_preview_func_aug",&preview);CHKERRQ(ierr);
    if (preview) {
      ierr          = PetscPrintf(PetscObjectComm((PetscObject)snes->vec_func_aug),"Initial augmented function vector:\n");CHKERRQ(ierr);
      ierr          = VecViewFromOptions(snes->vec_func_aug,((PetscObject)snes)->prefix,"-snes_newtonaugas_preview_func_aug");CHKERRQ(ierr);
    }
  }
  ierr          = VecZeroEntries(l);CHKERRQ(ierr);
  ierr          = SNESConstraintAugGather(snes,x_aug,x,l);CHKERRQ(ierr);
  ierr          = SNESNEWTONAUGASMeritFunction(snes,x_aug,&merit);CHKERRQ(ierr);
  ierr          = VecNorm(f,NORM_2,&fnorm);CHKERRQ(ierr);
  ierr          = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm    = fnorm;
  snes->merit   = merit;
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

    /* Check whether the solution is feasible. */
    ierr = SNESConstraintFindBoundsViolation(snes,x,&idx);CHKERRQ(ierr);
    if (idx >= 0) {
      PetscInt xlo;

      ierr = VecGetOwnershipRange(x,&xlo,NULL);CHKERRQ(ierr);
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Infeasible state: constraint %D is outside bounds",idx+xlo);
    }

    ierr = SNESConstraintComputeJacobians(snes,x,MAT_INITIAL_MATRIX,snes->jacobian_aug,snes->jacobian_aug_pre,snes->jacobian_aug_shape,snes->jacobian,snes->jacobian_pre,snes->jacobian_constr,snes->jacobian_constrt);CHKERRQ(ierr);
    new_active = NULL;
    ierr = SNESNEWTONAUGASInitialActiveSet_Private(snes,&active);CHKERRQ(ierr);
    do {
      if (new_active) { /* active set has been updated */
	    ierr = ISDestroy(&active);CHKERRQ(ierr);
	    active = new_active; new_active = NULL;
      }
      ierr = ISGetLocalSize(active,&nactive);CHKERRQ(ierr);
      ierr = ISGetIndices(active,&iactive);CHKERRQ(ierr);
      ierr = VecGetArray(l,&larray);CHKERRQ(ierr);
      for (i = 0; i < nactive;++i) larray[iactive[i]] = 0.0;
      ierr = VecRestoreArray(l,&larray);CHKERRQ(ierr);
      ierr = SNESNEWTONAUGASComputeSearchDirection_Private(snes,active,&lits);CHKERRQ(ierr);
      ierr = SNESNEWTONAUGASModifyActiveSet_Private(snes,active,&new_active,&tbar);CHKERRQ(ierr);
    } while (new_active);

    /* Compute a (scaled) negative update in the line search routine:
         x_aug <- x_aug - alpha*dx_aug
       and evaluate f = function(x) (depends on the line search).
    */
    /* FIXME: what is hnorm? */
    hnorm = fnorm;

    ierr = SNESConstraintAugGather(snes,x_aug,x,l);CHKERRQ(ierr);
    ierr = VecScale(x_aug,tbar);CHKERRQ(ierr);
    /* TODO: Jason,Todd, what should we be passing in for f_aug?  Anything? The below stuff is almost certainly incorrect. */
    ierr = SNESConstraintAugGather(snes,f_aug,f,NULL);CHKERRQ(ierr);
    ierr = SNESConstraintAugGather(snes,dx_aug,dx,dl);CHKERRQ(ierr);
    /* TODO: what can linesearch do with fnorm?  Should the merit function be passed in and out instead?
     Should we use whatever norm is stashed on the corresponding vector, whenever possible?
     The merit function, however, isn't cached anyplace convenient, since it's the norm of a
     transient vector.  We pass in NULL for now.
    */
    ierr = SNESLineSearchApply(linesearch,x_aug,f_aug,NULL,dx_aug);CHKERRQ(ierr);
    ierr = SNESConstraintAugScatter(snes,x_aug,x,l);CHKERRQ(ierr);
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
        /* PetscBool ismin=PETSC_FALSE; */
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        /*
	   TODO: replace this with a NEWTONAUGAS-specific check?

	   ierr         = SNESNEWTONLSCheckLocalMin_Private(snes,snes->jacobian,f,w,fnorm,&ismin);CHKERRQ(ierr);
	   if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
	*/
        break;
      }
    }
    if (dmsnes->ops->projectontoconstraints) {
      ierr = (*dmsnes->ops->projectontoconstraints)(snes,x,x,dmsnes->projectontoconstraintsctx);CHKERRQ(ierr);
      ierr = SNESConstraintAugGather(snes,x_aug,x,l);CHKERRQ(ierr);
      ierr = SNESNEWTONAUGASMeritFunction(snes,x_aug,&merit);CHKERRQ(ierr);

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
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,dxnorm,snes->merit,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == snes->max_its) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESActiveConstraints_Default"
PETSC_INTERN PetscErrorCode SNESActiveConstraints_Default(SNES snes,Vec x,IS *active,IS *basis,void *ctx)
{
  /* PetscErrorCode    ierr; */
  /* SNES_NEWTONAUGAS     *newtaugas = (SNES_NEWTONAUGAS*) snes->data; */

  PetscFunctionBegin;
  /* TODO: Apply QR or SVD to a redundant B? */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NEWTONAUGAS"
PETSC_INTERN PetscErrorCode SNESSetFromOptions_NEWTONAUGAS(SNES snes)
{
  PetscErrorCode    ierr;
  PetscViewer       monviewer;
  char              monfilename[PETSC_MAX_PATH_LEN+1];
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNESNEWTONAUGAS solver options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-snes_newtonaugas_monitor","Monitor active set merit function","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)snes),monfilename,&monviewer);CHKERRQ(ierr);
    ierr = SNESMonitorSet(snes,SNESNEWTONAUGASMonitorDefault,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset_NEWTONAUGAS"
PETSC_INTERN PetscErrorCode SNESReset_NEWTONAUGAS(SNES snes)
{
  /* TODO */
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NEWTONAUGAS"
PETSC_INTERN PetscErrorCode SNESSetUp_NEWTONAUGAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNESLineSearch    linesearch;
  SNES_NEWTONAUGAS  *newtaugas = (SNES_NEWTONAUGAS*) snes->data;
  PetscErrorCode    (*func)(SNES,Vec,Vec,void*), (*afunc)(SNES,Vec,Vec,void*);
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = SNESGetFunction(snes,NULL,&func,NULL);CHKERRQ(ierr);
  ierr = SNESConstraintGetAugFunction(snes,NULL,NULL,NULL,&afunc,NULL);CHKERRQ(ierr);
  /* TODO:
   The dependencies between the function calls below indicate that these call
   sequence should be hidden in two different setup functions: afunc-setup and func-setup.
   These are generic and belong at the SNESConstraintXXX() level.
   */
  if (afunc) {
    ierr = SNESConstraintSetUpAugVectors(snes);CHKERRQ(ierr);
    ierr = SNESConstraintSetUpAugEmbeddings(snes,&snes->is_func_aug,&snes->is_constr_aug);CHKERRQ(ierr); /* Depends on the existence of aug vec. */
    ierr = SNESConstraintSetUpSplitVectors(snes);CHKERRQ(ierr); /* Depends on the existence of embeddings. */
    ierr = SNESConstraintSetUpAugScatters(snes);CHKERRQ(ierr); /* Depends on the existence of aug and split vectors and embeddings. */
    ierr = SNESConstraintSetUpSplitBounds(snes);CHKERRQ(ierr);  /* Depends on the existence of scatters to extract split bounds. */

    ierr = SNESConstraintSetUpAugMatrices(snes);CHKERRQ(ierr);
    ierr = SNESConstraintSetUpSplitMatrices(snes);CHKERRQ(ierr); /* Depends on the existence of embeddings. */
  }
  if (func) {
    ierr = SNESConstraintSetUpSplitVectors(snes);CHKERRQ(ierr);
    ierr = SNESConstraintSetUpAugVectors(snes);CHKERRQ(ierr);    /* Depends on the existence of split vectors. */
    ierr = SNESConstraintSetUpAugEmbeddings(snes,&snes->is_func_aug,&snes->is_constr_aug);CHKERRQ(ierr);    /* Depends on the existence of aug and split vectors. */
    ierr = SNESConstraintSetUpScatters(snes);CHKERRQ(ierr);      /* Depends on the existence of aug and split vectors and embeddings. */

    ierr = SNESConstraintSetUpSplitMatrices(snes);CHKERRQ(ierr);
    /* SNESConstraintSetUpAugMatrices() is a no-op if !afunc. */
  }
  ierr = SNESConstraintSetWorkVecs(snes,SNES_NEWTONAUGAS_WORK_N,SNES_NEWTONAUGAS_WORK_CONSTR_N,SNES_NEWTONAUGAS_WORK_AUG_N);CHKERRQ(ierr);
  if (snes->vec_constrl) {
    ierr          = PetscOptionsHasName(((PetscObject)snes)->prefix,"-snes_newtonaugas_view_lower_bound",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr        = PetscPrintf(PetscObjectComm((PetscObject)snes->vec_constrl),"Lower bound:\n");CHKERRQ(ierr);
      ierr          = VecViewFromOptions(snes->vec_constrl,((PetscObject)snes)->prefix,"-snes_newtonaugas_view_lower_bound");CHKERRQ(ierr);
    }
  }
  if (snes->vec_constru) {
    ierr          = PetscOptionsHasName(((PetscObject)snes)->prefix,"-snes_newtonaugas_view_upper_bound",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr        = PetscPrintf(PetscObjectComm((PetscObject)snes->vec_constru),"Upper bound:\n");CHKERRQ(ierr);
      ierr          = VecViewFromOptions(snes->vec_constru,((PetscObject)snes)->prefix,"-snes_newtonaugas_view_upper_bound");CHKERRQ(ierr);
    }
  }
  /* TODO: handle absence of one or both bounds vectors. */

  ierr = VecDuplicate(snes->vec_constr,&newtaugas->vec_sol_lambda);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_constr,&newtaugas->vec_sol_lambda_update);CHKERRQ(ierr);

  /* Linear solver. */
  ierr = KSPCreate(PetscObjectComm((PetscObject)snes),&newtaugas->kspr_aug);CHKERRQ(ierr);
  ierr = PetscObjectPrependOptionsPrefix((PetscObject)newtaugas->kspr_aug,"newtonaugas_aug_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(newtaugas->kspr_aug);CHKERRQ(ierr);

  /* TODO!: set up the linesearch. */
  ierr = SNESGetLineSearch(snes,&linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBTSD);CHKERRQ(ierr);
  ierr = SNESLineSearchSetMerit(linesearch,SNESNEWTONAUGASMeritFunction);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NEWTONAUGAS"
PETSC_INTERN PetscErrorCode SNESDestroy_NEWTONAUGAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAUGAS     *newtaugas = (SNES_NEWTONAUGAS*) snes->data;

  PetscFunctionBegin;
  /* TODO: factor out some of this code into SNESReset_NEWTONAUGAS() */
  ierr = VecDestroy(&newtaugas->vec_sol_lambda);CHKERRQ(ierr);
  ierr = VecDestroy(&newtaugas->vec_sol_lambda_update);CHKERRQ(ierr);
  ierr = MatDestroy(&newtaugas->Br);CHKERRQ(ierr);
  ierr = MatDestroy(&newtaugas->Brt);CHKERRQ(ierr);
  ierr = MatDestroy(&newtaugas->Bb_pre);CHKERRQ(ierr);
  ierr = MatDestroy(&newtaugas->Bbt_pre);CHKERRQ(ierr);
  ierr = KSPDestroy(&newtaugas->kspr_aug);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESNEWTONAUGAS - Augmented-space active-set linesearch-based Newton-like solver
      for nonlinear problems with constraints (variational inequalities or mixed
      complementarity problems).

   Options Database:
.   -snes_newtonaugas_type primal|dual|saddle


   Level: beginner

   References:
   - T. S. Munson, and S. Benson. Flexible Complementarity Solvers for Large-Scale
     Applications, Optimization Methods and Software, 21 (2006).

.seealso:  SNES, SNESCreate(), SNESSetType(), SNESLineSearchSet(),
           SNESSetFunction(), SNESSetJacobian(), SNESConstraintSetFunction(), SNESConstraintSetJacobian(),
           SNESSetActiveConstraints(), SNESConstraintSetProjectOntoConstraints(), SNESSetDistanceToConstraints(),
           SNESNEWTONAUGASSetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NEWTONAUGAS"
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONAUGAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAUGAS     *newtaugas;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONAUGAS;
  snes->ops->destroy        = SNESDestroy_NEWTONAUGAS;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONAUGAS;

  ierr                = PetscNewLog(snes,&newtaugas);CHKERRQ(ierr);
  snes->data          = (void*)newtaugas;

  PetscFunctionReturn(0);
}


/*MC
    SNESNEWTONAUGASActiveConstraintBasis - callback function identifying a basis for the active constraints
    linearized at the current state vector x of the constrained nonlinear problem (variational inequality)
    solved by SNES

     Synopsis:
     #include <petscsnes.h>
     SNESNEWTONAUGASActiveConstraintBasis(SNES snes,Vec x,Vec f,Vec g,Vec B,IS active,IS *basis,Mat *Bb_pre,Mat *Bbt_pre,void *ctx);

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
     linearized constraint range. Since in primal elimination methods inverts (or solves with) of both Bb
     and Bbt, the transpose of Bb, are needed, the user can provide matrices to build preconditioners for
     both Bb and Bbt.


   Level: intermediate

.seealso:   SNESNEWTONAUGASSetAcitveConstraintBasis(), SNESNEWTONAUGASSetActiveConstraints(), SNESConstraintSetFunction(), SNESConstraintSetJacobian(), SNESConstraintFunction, SNESConstraintJacobian

 M*/

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONAUGASSetActiveConstraintBasis"
/*@C
   SNESNEWTONAUGASSetActiveConstraintBasis -   sets the callback identifying a basis for linearized active constraints.

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

.seealso: SNESNEWTONAUGASGetActiveConstraintBasiss(), SNESConstraintSetFunction(), SNESNEWTONAUGASActiveConstraintBasis
@*/
PetscErrorCode  SNESNEWTONAUGASSetActiveConstraintBasis(SNES snes,Mat Bb_pre,Mat Bbt_pre,PetscErrorCode (*f)(SNES,Vec,Vec,Vec,Mat,IS,IS*,Mat*,Mat*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  SNES_NEWTONAUGAS  *newtaugas = (SNES_NEWTONAUGAS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  /* TODO: Check whether this is a SNESNEWTONAUGAS */
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESNEWTONAUGASSetActiveConstraintBasis(dm,f,ctx);CHKERRQ(ierr);
  if (Bb_pre) {
    ierr = PetscObjectReference((PetscObject)Bb_pre);CHKERRQ(ierr);
    ierr = MatDestroy(&newtaugas->Bb_pre);CHKERRQ(ierr);
    newtaugas->Bb_pre = Bb_pre;
  }
  if (Bbt_pre) {
    ierr = PetscObjectReference((PetscObject)Bbt_pre);CHKERRQ(ierr);
    ierr = MatDestroy(&newtaugas->Bbt_pre);CHKERRQ(ierr);
    newtaugas->Bbt_pre = Bbt_pre;
  }
  PetscFunctionReturn(0);
}
