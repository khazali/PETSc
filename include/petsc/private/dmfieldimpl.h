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
  PetscErrorCode (*evaluate) (DMField,Vec,PetscDataType,void*,void*,void*);
  PetscErrorCode (*evaluateFECompact) (DMField,PetscInt,const PetscInt *,PetscQuadrature,PetscDataType,PetscBool,void*,PetscBool,void*,PetscBool,void*);
  PetscErrorCode (*evaluateFV) (DMField,PetscInt,const PetscInt *,PetscDataType,void*,void*,void*);
  PetscErrorCode (*getFEInvariance) (DMField,PetscInt,const PetscInt *,PetscBool*,PetscBool*,PetscBool*);
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
