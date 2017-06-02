#ifndef __PETSCGLL_H
#define __PETSCGLL_H
#include <petscsys.h>

/*S
    PetscGLLIP  - the locations and weights of the Gauss-Lobatto-Legendre nodes of a given size

  Level: beginner

  References: XXXX

.seealso: PetscGLLIPCreate(), PetscGLLIPDestroy(), PetscGLLIPView()
S*/
typedef struct {
  PetscInt    n;
  PetscReal   *nodes;
  PetscScalar *weights;
} PetscGLLIP;

/*E
  PetscGLLIPCreateType - algorithm used to compute the GLL nodes and weights

  Level: beginner

$  PETSCGLLIP_VIA_LINEARALGEBRA - compute the nodes via linear algebra
$  PETSCGLLIP_VIA_NEWTON - compute the nodes by solving a nonlinear equation with Newton's method

.seealso: PetscGLLIP, PetscGLLIPCreate()
E*/
typedef enum {PETSCGLLIP_VIA_LINEARALGEBRA,PETSCGLLIP_VIA_NEWTON} PetscGLLIPCreateType;

#endif
PETSC_EXTERN PetscErrorCode PetscGLLIPCreate(PetscInt,PetscGLLIPCreateType,PetscGLLIP*);
PETSC_EXTERN PetscErrorCode PetscGLLIPDestroy(PetscGLLIP*);
PETSC_EXTERN PetscErrorCode PetscGLLIPView(PetscGLLIP*,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscGLLIPElementStiffnessCreate(PetscGLLIP*,PetscReal***);
PETSC_EXTERN PetscErrorCode PetscGLLIPElementStiffnessDestroy(PetscGLLIP*,PetscReal***);
PETSC_EXTERN PetscErrorCode PetscGLLIPMassCreate(PetscGLLIP*,PetscReal***);
PETSC_EXTERN PetscErrorCode PetscGLLIPMassDestroy(PetscGLLIP*,PetscReal***);
PETSC_EXTERN PetscErrorCode PetscGLLIPIntegrate(PetscGLLIP*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscGLLIPElementGradientCreate(PetscGLLIP*,PetscReal***,PetscReal***);
PETSC_EXTERN PetscErrorCode PetscGLLIPElementGradientDestroy(PetscGLLIP*,PetscReal***,PetscReal***);
PETSC_EXTERN PetscErrorCode PetscGLLIPElementAdvectionCreate(PetscGLLIP*,PetscReal***);
PETSC_EXTERN PetscErrorCode PetscGLLIPElementAdvectionDestroy(PetscGLLIP*,PetscReal***);
