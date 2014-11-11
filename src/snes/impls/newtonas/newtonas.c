
#include <../src/snes/impls/newtonas/newtonasimpl.h> /*I "petscsnes.h" I*/
#include <petsc-private/kspimpl.h>
#include <petsc-private/matimpl.h>
#include <petsc-private/dmimpl.h>
#include <petsc-private/vecimpl.h>

/* QUESTIONS:
   - How do we monitor convergence?  Need a monitor routine.
   - How do we ensure that in the absence of true constraints (i.e., when all constraints are +/-Inf)
     the algorithm reduces to an efficient approximation of constraint-free SNES (e.g., similar to SNESNEWTONLS).
*/


/* SNES NEWTONAS ALGORITHM SUBROUTINE STUBS BEGIN */

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASInitialActiveSet_Private"
static PetscErrorCode SNESNEWTONASInitialActiveSet_Private(SNES snes,Vec x,Vec l,Vec f,Vec g,IS *active)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* TODO: implement */
  /* A = \{i : (g_i(x) = g_i^l & \lambda_i > 0) | (g_i(x) = g_i^u & \lambda_i < 0)\} -- strongly active set. */
  /* TODO: check signs on \lambda in the foregoing def.*/
  /*
    This should use the distance to boundary computed in SNESNEWTONASComputeDistanceToBoundary().
     Should we just roll that subroutine into this one?  It's not used anywhere else.
  */
  *active = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASComputeDistanceToBoundary"
static PetscErrorCode SNESNEWTONASComputeDistanceToBoundary(SNES snes,Vec x,Vec l,Vec dx,Vec dl,Mat B,Vec distg,Vec distl)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* TODO: implement */
  /* ||f - l*B||_2^2  */
  /* N.B.: this function might need to set domain error, when necessary (e.g., when computing f results in a domain error?) */
  *merit = 0.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASModifyActiveSet_Private"
static PetscErrorCode SNESNEWTONASModifyActiveSet_Private(SNES snes,Vec x,Vec l,Vec f,Vec g,Vec dx,Vec dl,IS active,IS *new_active,PetscReal *tbar)
{
  /* Returns a modified IS based on the distance to constraint bounds, or NULL if no modification is necessary. */
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* TODO: implement */
  /*
     Also compute \overline t -- the upper bound on the search step size. This is because upper bounds on the individual
     constraints are examined here anyway.  tbar is only one _MPI reduction_ away.
  */
  *new_active = NULL;
  *tbar = 0.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASComputeMeritFunctionDefault"
static PetscErrorCode SNESNEWTONASComputeMeritFunctionDefault(SNES snes,Vec x,Vec l,Vec f,Vec g,Mat B,PetscReal *merit)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* TODO: implement */
  /* ||f - l*B||_2^2  */
  /* N.B.: this function might need to set domain error, when necessary (e.g., when computing f results in a domain error?) */
  *merit = 0.0;
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASComputeSearchDirectionPrimal_Private"
static PetscErrorCode SNESNEWTONASComputeSearchDirectionPrimal_Private(SNES snes,Vec x,Vec l,Vec f,Mat A,Mat Apre,Mat B,IS active,Vec dx,Vec dl)
{
  /* Observe that only a subvector of dl defined by the active set is nonzero. */
  PetscErrorCode     ierr;
  SNES_NEWTONAS      *newtas = (SNES_NEWTONAS*)(snes->data);
  Vec                tildedl; /* \tilde \delta \lambda */
  Vec                q = snes->work[2];   /* q = -(f - B^T*l) */

  PetscFunctionBegin;
#if 0
  /* The following code needs to be cleaned up so that it at least compiles. */
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

/* Also need two line searches: Armijo and projection-based. */
/* SNES NEWTONAS ALGORITHM SUBROUTINE STUBS BEGIN */


#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NEWTONAS"
PetscErrorCode SNESSolve_NEWTONAS(SNES snes)
{
  PetscErrorCode     ierr;
  Vec                X,dX,F,H,L,dL,G,W;
  SNES_NEWTONAS      *newtas = (SNES_NEWTONAS*)snes->data;
  DM                  dm;
  DMSNES              dmsnes;
  PetscInt            i,lits;
  PetscBool           lssucceed;
  PetscReal           fnorm,hnorm,xnorm,dxnorm;
  KSPConvergedReason  kspreason;
  PetscBool           domainerror;
  SNESLineSearch      linesearch;
  SNESConvergedReason reason;
  IS                  active,new_active;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  x      = snes->vec_sol;               /* solution vector */
  f      = snes->vec_func;              /* residual vector */
  dx     = snes->vec_sol_update;        /* newton step */
  g      = snes->vec_constr;            /* constraints */
  l      = newtas->lambda[0];           /* \lambda */
  dl     = newtas->lambda[1];           /* \delta \lambda */
  h      = snes->work[0];               /* residual at the linesearch location */
  w      = snes->work[1];               /* linear update at the linesearch location */


  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDMSNES(dm,&dmsnes);CHKERRQ(ierr);

  /*
     FIXME: Nonlinear preconditioning would go here.
  */

  if (dmsnes->ops->projectontoconstraints) {
    ierr = (*dmsnes->ops->projectontoconstraints)(snes,X,X,dmsnes->projectontoconstraintsctx);CHKERRQ(ierr);
  } /* No 'else' clause since there is really no default way of projecting onto constraints that I know of. */


  /* FIXME: replace this with an application of the merit function. We might need a flag analogous to snes->vec_func_init_set.  For relative error checking? */
  /* QUESTION: Do we need if(!snes->vec_func_init_set) wrapping the following block? See SNESSolve_NEWTONLS() */
  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
  } else snes->vec_func_init_set = PETSC_FALSE;

  /*
     QUESTION: How do we check convergence in the case with general constraints?
     The following might be a bad way to do it. Do we need to use a MERIT function?
     And/or dmsnes->ops->computeobjective?

     Convergence is either the merit function is equal to zero (you have solved the problem) or the norm of
     the gradient of the merit function is zero (you have a local minimizer of the merit function, but have
     NOT solved the complementarity problem).
   */

  /* FIXME: use the merit function.  How? */
  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);        /* fnorm <- ||F||  */
  if (PetscIsInfOrNanReal(fnorm)) {
    snes->reason = SNES_DIVERGED_FNORM_NAN;
    PetscFunctionReturn(0);
  }

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);
  /* test convergence */
  /* FIXME: use merit function */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<snes->max_its; ++i) {
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    /*
       FIXME: nonlinear PC application would go here.
    */

    /* Compute the primal and constraint Jacobians to form the reduced linear system below. */
    ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    /*
       FIXME: eventually we want something like SNESComputeConstraintJacobian(),
       which will handle the MF case of the constraint Jacobian, comparison to the
       explicitly-computed operator and debugging.
    */
    ierr = dmsnes->ops->constraintjacobian(snes,X,snes->jacobian_constr,dmsnes->constraintjacobianctx);CHKERRQ(ierr);

    /* TODO: compute the initial 'active'. */
    new_active = NULL;
    ierr = SNESNEWTONASInitialActiveSet_Private(snes,X,L,F,G,&active);CHKERRQ(ierr);
    do {
      PetscReal tbar;
      if (new_active) { /* active set has been updated */
	    ierr = ISDestroy(&active);CHKERRQ(ierr);
	    active = new_active; new_active = NULL;
      }
      if (newtas->type == SNES_NEWTONAS_PRIMAL) {
	ierr = SNESNEWTONASComputeSearchDirectionPrimal_Private(snes,X,L,F,snes->jacobian,snes->jacobian_pre,snes->jacobian_constr,active,dX,dL);CHKERRQ(ierr);
      }
      else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"SNESNEWTONAS solver type not yet supported: %s",type);
      ierr = SNESNEWTONASModifyActiveSet_Private(snes,X,L,F,G,dX,dL,active,&new_active,&tbar);CHKERRQ(ierr);
    } while (new_active);


    /* TODO: from here to the end of the subroutine the code needs to be updated to take constraints into account. */
    /* Compute a (scaled) negative update in the line search routine:
         X <- X - alpha*dX
       and evaluate F = function(X) (depends on the line search).
    */
    hnorm = fnorm;
    ierr  = SNESLineSearchApply(linesearch, X, F, &fnorm, dX);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetSuccess(linesearch, &lssucceed);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &dxnorm);CHKERRQ(ierr);
    ierr  = PetscInfo4(snes,"fnorm=%18.16e, hnorm=%18.16e, dxnorm=%18.16e, lssucceed=%d\n",(double)hnorm,(double)fnorm,(double)dxnorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;

    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (snes->stol*xnorm > dxnorm) {
        snes->reason = SNES_CONVERGED_SNORM_RELATIVE;
        PetscFunctionReturn(0);
      }
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        /*
	   FIXME: replace this with a NEWTONAS-specific check?
	   ierr         = SNESNEWTONLSCheckLocalMin_Private(snes,snes->jacobian,F,W,fnorm,&ismin);CHKERRQ(ierr);
	*/
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,dxnorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  newtas->type = type;
  snes->ops->solve = SNESSolve_NEWTONAS_Primal;

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
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  if (type) *type = newtas->type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESActiveConstraints_Default"
PETSC_INTERN PetscErrorCode SNESActiveConstraints_Default(SNES snes,Vec x,IS *active,IS *basis,void *ctx)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

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

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNESNEWTONAS solver options");CHKERRQ(ierr);
  newtas->type = SNESNEWTONAS_PRIMAL;
  ierr = PetscOptionsEnum("-snes_newtonas_type","Type of linear solver to use for the saddle-point problem","SNESNEWTONASSetType",SNESNEWTONASTypes,(PetscEnum)newtas->type,(PetscEnum*)&newtas->type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NEWTONAS"
PETSC_INTERN PetscErrorCode SNESSetUp_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  if (!newtas->type) {
    ierr = SNESNEWTONASSetType(snes,SNESNEWTONAS_PRIMAL);CHKERRQ(ierr);
  }
  if (newtas->lambda) {
    ierr = VecDestroyVecs(2,&newtas->lambda);CHKERRQ(ierr);
  }
  /*
     QUESTION: Do we need to go through the public API to set private data structures?
     The rationale given in the docs is that this gives control to plugins, but wouldn't
     they have to go into the SNES data structures to make use of the work vecs?
     A related question: why not let the impl allocated and clean up its own work vecs?
  */
  ierr = SNESSetWorkVecs(snes,3);CHKERRQ(ierr);
  if (!newtas->lambda) {
    ierr = PetscMalloc(2*sizeof(Vec),&newtas->lambda);CHKERRQ(ierr);
  }
  ierr = VecDuplicateVecs(snes->vec_constr,2,&newtas->lambda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NEWTONAS"
PETSC_INTERN PetscErrorCode SNESDestroy_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  /* TODO: Tear things down. */
  if (newtas->lambda) {
    ierr = VecDestroyVecs(2,&newtas->lambda);CHKERRQ(ierr);
  }
  ierr = PetscFree(newtas->lambda);CHKERRQ(ierr);
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

/*MC
    SNESNEWTONASActiveConstraints - callback function identifying the active constraints
    at the current state vector x of the constrained nonlinear problem (variational inequality)
    solved by SNESNEWTONAS

     Synopsis:
     #include <petscsnes.h>
     SNESNEWTONASActiveConstraints(SNES snes,Vec x,Vec f,Vec g,Mat B,IS *active,void *ctx);

     Input Parameters:
+     snes - the SNES context
.     x    - state at which to evaluate activities
.     f    - function at x
.     g    - constraints at x
.     B    - constraint Jacobian at x
-     ctx  - optional user-defined function context, passed in with SNESSetActiveConstraints()

     Output Parameters:
.     active  - indices of active constraints


     Notes:
     Active constraints are essentially those that would be violated when moving along the direction of
     the SNES function f.  The linearized constraints are the span of the rows of the constraint Jacobian B.
     Active constraints (linearized or otherwise) are labled by the corresponding rows of the constraint
     Jacobian.  The active constraint Jacobian is the submatrix B of the constraint Jacobian comprising
     the active constraint rows. Output parameter 'active' is exactly the indices of the active Jacobian
     rows.

   Level: intermediate

.seealso:   SNESNEWTONASSetAcitveConstraints(), SNESSetConstraintFunction(), SNESSetConstraintJacobian(), SNESConstraintFunction, SNESConstraintJacobian

 M*/


#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASSetActiveConstraints"
/*@C
   SNESNEWTONASSetActiveConstraints -   sets the callback identifying active constraints.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  f    - function identifying the active constraints at the current state x
-  ctx  - optional (if not NULL) user-defined context for private data for the
          identification of active constraints function.

   Level: intermediate

.keywords: SNES, nonlinear, set, active, constraint, function

.seealso: SNESNEWTONASGetActiveConstraints(), SNESSetConstraintFunction(), SNESNEWTONASActiveConstraints
@*/
PetscErrorCode  SNESNEWTONASSetActiveConstraints(SNES snes,PetscErrorCode (*f)(SNES,Vec,Vec,Vec,Mat,IS*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESNEWTONASSetActiveConstraints(dm,f,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNEWTONASGetActiveConstraints"
/*@C
   SNESNEWTONASGetActiveConstraints -   retrieves the callback identifying active constraints.

   Logically Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Output Parameters:
.  f    - function identifying the active constraints at the current state x
-  ctx  - optional (if not NULL) user-defined context for private data for the
          identification of active constraints function.

   Level: intermediate

.keywords: SNES, nonlinear, get, active, constraint, function

.seealso: SNESNEWTONASSetActiveConstraints(), SNESSetConstraintFunction(), SNESNEWTONASActiveConstraints
@*/
PetscErrorCode  SNESNEWTONASGetActiveConstraints(SNES snes,PetscErrorCode (**f)(SNES,Vec,Vec,Vec,Mat,IS*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESNEWTONASGetActiveConstraints(dm,f,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
