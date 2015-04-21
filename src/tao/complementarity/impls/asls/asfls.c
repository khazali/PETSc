#include <../src/tao/complementarity/impls/ssls/ssls.h>
/*
   Context for ASXLS
     -- active-set      - reduced matrices formed
                          - inherit properties of original system
     -- semismooth (S)  - function not differentiable
                        - merit function continuously differentiable
                        - Fischer-Burmeister reformulation of complementarity
                          - Billups composition for two finite bounds
     -- infeasible (I)  - iterates not guaranteed to remain within bounds
     -- feasible (F)    - iterates guaranteed to remain within bounds
     -- linesearch (LS) - Armijo rule on direction

   Many other reformulations are possible and combinations of
   feasible/infeasible and linesearch/trust region are possible.

   Basic theory
     Fischer-Burmeister reformulation is semismooth with a continuously
     differentiable merit function and strongly semismooth if the F has
     lipschitz continuous derivatives.

     Every accumulation point generated by the algorithm is a stationary
     point for the merit function.  Stationary points of the merit function
     are solutions of the complementarity problem if
       a.  the stationary point has a BD-regular subdifferential, or
       b.  the Schur complement F'/F'_ff is a P_0-matrix where ff is the
           index set corresponding to the free variables.

     If one of the accumulation points has a BD-regular subdifferential then
       a.  the entire sequence converges to this accumulation point at
           a local q-superlinear rate
       b.  if in addition the reformulation is strongly semismooth near
           this accumulation point, then the algorithm converges at a
           local q-quadratic rate.

   The theory for the feasible version follows from the feasible descent
   algorithm framework.

   References:
     Billups, "Algorithms for Complementarity Problems and Generalized
       Equations," Ph.D thesis, University of Wisconsin - Madison, 1995.
     De Luca, Facchinei, Kanzow, "A Semismooth Equation Approach to the
       Solution of Nonlinear Complementarity Problems," Mathematical
       Programming, 75, pages 407-439, 1996.
     Ferris, Kanzow, Munson, "Feasible Descent Algorithms for Mixed
       Complementarity Problems," Mathematical Programming, 86,
       pages 475-497, 1999.
     Fischer, "A Special Newton-type Optimization Method," Optimization,
       24, pages 269-284, 1992
     Munson, Facchinei, Ferris, Fischer, Kanzow, "The Semismooth Algorithm
       for Large Scale Complementarity Problems," Technical Report 99-06,
       University of Wisconsin - Madison, 1999.
*/


#undef __FUNCT__
#define __FUNCT__ "TaoSetUp_ASFLS"
PetscErrorCode TaoSetUp_ASFLS(Tao tao)
{
  TAO_SSLS       *asls = (TAO_SSLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&asls->ff);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&asls->dpsi);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&asls->da);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&asls->db);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&asls->t1);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&asls->t2);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &asls->w);CHKERRQ(ierr);
  asls->fixed = NULL;
  asls->free = NULL;
  asls->J_sub = NULL;
  asls->Jpre_sub = NULL;
  asls->r1 = NULL;
  asls->r2 = NULL;
  asls->r3 = NULL;
  asls->dxfree = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Tao_ASLS_FunctionGradient"
static PetscErrorCode Tao_ASLS_FunctionGradient(TaoLineSearch ls, Vec X, PetscReal *fcn,  Vec G, void *ptr)
{
  Tao            tao = (Tao)ptr;
  TAO_SSLS       *asls = (TAO_SSLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoComputeConstraints(tao, X, tao->constraints);CHKERRQ(ierr);
  ierr = VecFischer(X,tao->constraints,tao->XL,tao->XU,asls->ff);CHKERRQ(ierr);
  ierr = VecNorm(asls->ff,NORM_2,&asls->merit);CHKERRQ(ierr);
  *fcn = 0.5*asls->merit*asls->merit;
  ierr = TaoComputeJacobian(tao,tao->solution,tao->jacobian,tao->jacobian_pre);CHKERRQ(ierr);

  ierr = MatDFischer(tao->jacobian, tao->solution, tao->constraints,tao->XL, tao->XU, asls->t1, asls->t2,asls->da, asls->db);CHKERRQ(ierr);
  ierr = VecPointwiseMult(asls->t1, asls->ff, asls->db);CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->jacobian,asls->t1,G);CHKERRQ(ierr);
  ierr = VecPointwiseMult(asls->t1, asls->ff, asls->da);CHKERRQ(ierr);
  ierr = VecAXPY(G,1.0,asls->t1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDestroy_ASFLS"
static PetscErrorCode TaoDestroy_ASFLS(Tao tao)
{
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&ssls->ff);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->dpsi);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->da);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->db);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->w);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->t1);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->t2);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->r1);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->r2);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->r3);CHKERRQ(ierr);
  ierr = VecDestroy(&ssls->dxfree);CHKERRQ(ierr);
  ierr = MatDestroy(&ssls->J_sub);CHKERRQ(ierr);
  ierr = MatDestroy(&ssls->Jpre_sub);CHKERRQ(ierr);
  ierr = ISDestroy(&ssls->fixed);CHKERRQ(ierr);
  ierr = ISDestroy(&ssls->free);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  tao->data = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolve_ASFLS"
static PetscErrorCode TaoSolve_ASFLS(Tao tao)
{
  TAO_SSLS                     *asls = (TAO_SSLS *)tao->data;
  PetscReal                    psi,ndpsi, normd, innerd, t=0;
  PetscInt                     nf;
  PetscErrorCode               ierr;
  TaoConvergedReason           reason;
  TaoLineSearchConvergedReason ls_reason;

  PetscFunctionBegin;
  /* Assume that Setup has been called!
     Set the structure for the Jacobian and create a linear solver. */

  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch,Tao_ASLS_FunctionGradient,tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetObjectiveRoutine(tao->linesearch,Tao_SSLS_Function,tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);

  ierr = VecMedian(tao->XL, tao->solution, tao->XU, tao->solution);CHKERRQ(ierr);

  /* Calculate the function value and fischer function value at the
     current iterate */
  ierr = TaoLineSearchComputeObjectiveAndGradient(tao->linesearch,tao->solution,&psi,asls->dpsi);CHKERRQ(ierr);
  ierr = VecNorm(asls->dpsi,NORM_2,&ndpsi);CHKERRQ(ierr);

  while (1) {
    /* Check the converged criteria */
    ierr = PetscInfo3(tao,"iter %D, merit: %g, ||dpsi||: %g\n",tao->niter,(double)asls->merit,(double)ndpsi);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,asls->merit,ndpsi,0.0,t,&reason);CHKERRQ(ierr);
    tao->niter++;
    if (TAO_CONTINUE_ITERATING != reason) break;

    /* We are going to solve a linear system of equations.  We need to
       set the tolerances for the solve so that we maintain an asymptotic
       rate of convergence that is superlinear.
       Note: these tolerances are for the reduced system.  We really need
       to make sure that the full system satisfies the full-space conditions.

       This rule gives superlinear asymptotic convergence
       asls->atol = min(0.5, asls->merit*sqrt(asls->merit));
       asls->rtol = 0.0;

       This rule gives quadratic asymptotic convergence
       asls->atol = min(0.5, asls->merit*asls->merit);
       asls->rtol = 0.0;

       Calculate a free and fixed set of variables.  The fixed set of
       variables are those for the d_b is approximately equal to zero.
       The definition of approximately changes as we approach the solution
       to the problem.

       No one rule is guaranteed to work in all cases.  The following
       definition is based on the norm of the Jacobian matrix.  If the
       norm is large, the tolerance becomes smaller. */
    ierr = MatNorm(tao->jacobian,NORM_1,&asls->identifier);CHKERRQ(ierr);
    asls->identifier = PetscMin(asls->merit, 1e-2) / (1 + asls->identifier);

    ierr = VecSet(asls->t1,-asls->identifier);CHKERRQ(ierr);
    ierr = VecSet(asls->t2, asls->identifier);CHKERRQ(ierr);

    ierr = ISDestroy(&asls->fixed);CHKERRQ(ierr);
    ierr = ISDestroy(&asls->free);CHKERRQ(ierr);
    ierr = VecWhichBetweenOrEqual(asls->t1, asls->db, asls->t2, &asls->fixed);CHKERRQ(ierr);
    ierr = ISComplementVec(asls->fixed,asls->t1, &asls->free);CHKERRQ(ierr);

    ierr = ISGetSize(asls->fixed,&nf);CHKERRQ(ierr);
    ierr = PetscInfo1(tao,"Number of fixed variables: %D\n", nf);CHKERRQ(ierr);

    /* We now have our partition.  Now calculate the direction in the
       fixed variable space. */
    ierr = TaoVecGetSubVec(asls->ff, asls->fixed, tao->subset_type, 0.0, &asls->r1);CHKERRQ(ierr);
    ierr = TaoVecGetSubVec(asls->da, asls->fixed, tao->subset_type, 1.0, &asls->r2);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(asls->r1,asls->r1,asls->r2);CHKERRQ(ierr);
    ierr = VecSet(tao->stepdirection,0.0);CHKERRQ(ierr);
    ierr = VecISAXPY(tao->stepdirection, asls->fixed, 1.0,asls->r1);CHKERRQ(ierr);

    /* Our direction in the Fixed Variable Set is fixed.  Calculate the
       information needed for the step in the Free Variable Set.  To
       do this, we need to know the diagonal perturbation and the
       right hand side. */

    ierr = TaoVecGetSubVec(asls->da, asls->free, tao->subset_type, 0.0, &asls->r1);CHKERRQ(ierr);
    ierr = TaoVecGetSubVec(asls->ff, asls->free, tao->subset_type, 0.0, &asls->r2);CHKERRQ(ierr);
    ierr = TaoVecGetSubVec(asls->db, asls->free, tao->subset_type, 1.0, &asls->r3);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(asls->r1,asls->r1, asls->r3);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(asls->r2,asls->r2, asls->r3);CHKERRQ(ierr);

    /* r1 is the diagonal perturbation
       r2 is the right hand side
       r3 is no longer needed

       Now need to modify r2 for our direction choice in the fixed
       variable set:  calculate t1 = J*d, take the reduced vector
       of t1 and modify r2. */

    ierr = MatMult(tao->jacobian, tao->stepdirection, asls->t1);CHKERRQ(ierr);
    ierr = TaoVecGetSubVec(asls->t1,asls->free,tao->subset_type,0.0,&asls->r3);CHKERRQ(ierr);
    ierr = VecAXPY(asls->r2, -1.0, asls->r3);CHKERRQ(ierr);

    /* Calculate the reduced problem matrix and the direction */
    ierr = TaoMatGetSubMat(tao->jacobian, asls->free, asls->w, tao->subset_type,&asls->J_sub);CHKERRQ(ierr);
    if (tao->jacobian != tao->jacobian_pre) {
      ierr = TaoMatGetSubMat(tao->jacobian_pre, asls->free, asls->w, tao->subset_type, &asls->Jpre_sub);CHKERRQ(ierr);
    } else {
      ierr = MatDestroy(&asls->Jpre_sub);CHKERRQ(ierr);
      asls->Jpre_sub = asls->J_sub;
      ierr = PetscObjectReference((PetscObject)(asls->Jpre_sub));CHKERRQ(ierr);
    }
    ierr = MatDiagonalSet(asls->J_sub, asls->r1,ADD_VALUES);CHKERRQ(ierr);
    ierr = TaoVecGetSubVec(tao->stepdirection, asls->free, tao->subset_type, 0.0, &asls->dxfree);CHKERRQ(ierr);
    ierr = VecSet(asls->dxfree, 0.0);CHKERRQ(ierr);

    /* Calculate the reduced direction.  (Really negative of Newton
       direction.  Therefore, rest of the code uses -d.) */
    ierr = KSPReset(tao->ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(tao->ksp, asls->J_sub, asls->Jpre_sub);CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp, asls->r2, asls->dxfree);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&tao->ksp_its);CHKERRQ(ierr);
    tao->ksp_tot_its+=tao->ksp_its;

    /* Add the direction in the free variables back into the real direction. */
    ierr = VecISAXPY(tao->stepdirection, asls->free, 1.0,asls->dxfree);CHKERRQ(ierr);


    /* Check the projected real direction for descent and if not, use the negative
       gradient direction. */
    ierr = VecCopy(tao->stepdirection, asls->w);CHKERRQ(ierr);
    ierr = VecScale(asls->w, -1.0);CHKERRQ(ierr);
    ierr = VecBoundGradientProjection(asls->w, tao->solution, tao->XL, tao->XU, asls->w);CHKERRQ(ierr);
    ierr = VecNorm(asls->w, NORM_2, &normd);CHKERRQ(ierr);
    ierr = VecDot(asls->w, asls->dpsi, &innerd);CHKERRQ(ierr);

    if (innerd >= -asls->delta*PetscPowReal(normd, asls->rho)) {
      ierr = PetscInfo1(tao,"Gradient direction: %5.4e.\n", (double)innerd);CHKERRQ(ierr);
      ierr = PetscInfo1(tao, "Iteration %D: newton direction not descent\n", tao->niter);CHKERRQ(ierr);
      ierr = VecCopy(asls->dpsi, tao->stepdirection);CHKERRQ(ierr);
      ierr = VecDot(asls->dpsi, tao->stepdirection, &innerd);CHKERRQ(ierr);
    }

    ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
    innerd = -innerd;

    /* We now have a correct descent direction.  Apply a linesearch to
       find the new iterate. */
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch, 1.0);CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &psi,asls->dpsi, tao->stepdirection, &t, &ls_reason);CHKERRQ(ierr);
    ierr = VecNorm(asls->dpsi, NORM_2, &ndpsi);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*MC
   TAOASFLS - Active-set feasible linesearch algorithm for solving
       complementarity constraints

   Options Database Keys:
+ -tao_ssls_delta - descent test fraction
- -tao_ssls_rho - descent test power

   Level: beginner
M*/
#undef __FUNCT__
#define __FUNCT__ "TaoCreate_ASFLS"
PETSC_EXTERN PetscErrorCode TaoCreate_ASFLS(Tao tao)
{
  TAO_SSLS       *asls;
  PetscErrorCode ierr;
  const char     *armijo_type = TAOLINESEARCHARMIJO;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,&asls);CHKERRQ(ierr);
  tao->data = (void*)asls;
  tao->ops->solve = TaoSolve_ASFLS;
  tao->ops->setup = TaoSetUp_ASFLS;
  tao->ops->view = TaoView_SSLS;
  tao->ops->setfromoptions = TaoSetFromOptions_SSLS;
  tao->ops->destroy = TaoDestroy_ASFLS;
  tao->subset_type = TAO_SUBSET_SUBVEC;
  asls->delta = 1e-10;
  asls->rho = 2.1;
  asls->fixed = NULL;
  asls->free = NULL;
  asls->J_sub = NULL;
  asls->Jpre_sub = NULL;
  asls->w = NULL;
  asls->r1 = NULL;
  asls->r2 = NULL;
  asls->r3 = NULL;
  asls->t1 = NULL;
  asls->t2 = NULL;
  asls->dxfree = NULL;
  asls->identifier = 1e-5;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, armijo_type);CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);

  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  tao->max_it = 2000;
  tao->max_funcs = 4000;
  tao->fatol = 0;
  tao->frtol = 0;
  tao->gttol = 0;
  tao->grtol = 0;
#if defined(PETSC_USE_REAL_SINGLE)
  tao->gatol = 1.0e-6;
  tao->fmin = 1.0e-4;
#else
  tao->gatol = 1.0e-16;
  tao->fmin = 1.0e-8;
#endif
  PetscFunctionReturn(0);
}

