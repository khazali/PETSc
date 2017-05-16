#if !defined(__DMFIELDIMPL_H)
#define      __DMFIELDIMPL_H

#include <petscdm.h>
#include <petscdmfield.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      DMFieldRegisterAllCalled;
PETSC_EXTERN PetscErrorCode DMFieldRegisterAll(void);

typedef struct _DMFieldOps *DMFieldOps;
struct _DMFieldOps {
  PetscErrorCode (*create) (DMField);
  PetscErrorCode (*destroy) (DMField);
  PetscErrorCode (*setfromoptions) (PetscOptionItems*,DMField);
  PetscErrorCode (*setup) (DMField);
  PetscErrorCode (*view) (DMField,PetscViewer);
  PetscErrorCode (*evaluate) (DMField,Vec,PetscScalar*,PetscScalar*,PetscScalar*);
  PetscErrorCode (*evaluateReal) (DMField,Vec,PetscReal*,PetscReal*,PetscReal*);
  PetscErrorCode (*evaluateFE) (DMField,PetscInt,const PetscInt *,PetscQuadrature,PetscScalar*,PetscScalar*,PetscScalar*);
  PetscErrorCode (*evaluateFEReal) (DMField,PetscInt,const PetscInt *,PetscQuadrature,PetscReal*,PetscReal*,PetscReal*);
  PetscErrorCode (*evaluateFV) (DMField,PetscInt,const PetscInt *,PetscScalar*,PetscScalar*,PetscScalar*);
  PetscErrorCode (*evaluateFVReal) (DMField,PetscInt,const PetscInt *,PetscReal*,PetscReal*,PetscReal*);
};
struct _p_DMField {
  PETSCHEADER(struct _DMFieldOps);
  DM dm;
  DMFieldContinuity continuity;
  PetscInt numComponents;
  void *data;
};

PetscErrorCode DMFieldCreate(DM,PetscInt,DMFieldContinuity,DMField*);
#endif
