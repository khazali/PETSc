#ifndef __TSOBJIMPL_H
#define __TSOBJIMPL_H
#include <petsc/private/tsimpl.h>

struct _TSObj {
  TSEvalObjective         f;         /* f(x,m,t) */
  TSEvalObjectiveGradient f_x;       /* \frac{\partial f}{\partial x}(x,m,t) */
  TSEvalObjectiveGradient f_m;       /* \frac{\partial f}{\partial m}(x,m,t) */
  Mat                     f_XX;
  TSEvalObjectiveHessian  f_xx;      /* \frac{\partial^2 f}{\partial x^2} f (x,m,t) */
  Mat                     f_XM;
  TSEvalObjectiveHessian  f_xm;      /* \frac{\partial^2 f}{\partial x \partial m} f (x,m,t) */
  Mat                     f_MM;
  TSEvalObjectiveHessian  f_mm;      /* \frac{\partial^2 f}{\partial m^2} f (x,m,t) */
  void                    *f_ctx;
  PetscReal               fixedtime; /* if the functional has to be evaluated at a specific time, i.e. || x(T1) - x_d || ^2, T1 in (T0,TF] */
  TSObj                   next;
};

PETSC_INTERN PetscErrorCode TSObjEval(TSObj,Vec,Vec,PetscReal,PetscReal*);
PETSC_INTERN PetscErrorCode TSObjEvalFixed(TSObj,Vec,Vec,PetscReal,PetscReal*);
PETSC_INTERN PetscErrorCode TSObjEval_U(TSObj,Vec,Vec,PetscReal,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEvalFixed_U(TSObj,Vec,Vec,PetscReal,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEval_M(TSObj,Vec,Vec,PetscReal,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEvalFixed_M(TSObj,Vec,Vec,PetscReal,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEval_UU(TSObj,Vec,Vec,PetscReal,Vec,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEvalFixed_UU(TSObj,Vec,Vec,PetscReal,Vec,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEval_UM(TSObj,Vec,Vec,PetscReal,Vec,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEvalFixed_UM(TSObj,Vec,Vec,PetscReal,Vec,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEval_MU(TSObj,Vec,Vec,PetscReal,Vec,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEvalFixed_MU(TSObj,Vec,Vec,PetscReal,Vec,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEval_MM(TSObj,Vec,Vec,PetscReal,Vec,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSObjEvalFixed_MM(TSObj,Vec,Vec,PetscReal,Vec,Vec,PetscBool*,Vec);
PETSC_INTERN PetscErrorCode TSGetNumObjectives(TS,PetscInt*);
PETSC_INTERN PetscErrorCode TSHasObjectiveIntegrand(TS,PetscBool*,PetscBool*,PetscBool*,PetscBool*,PetscBool*,PetscBool*);
PETSC_INTERN PetscErrorCode TSHasObjectiveFixed(TS,PetscReal,PetscReal,PetscBool*,PetscBool*,PetscBool*,PetscBool*,PetscBool*,PetscBool*,PetscReal*);
#endif
