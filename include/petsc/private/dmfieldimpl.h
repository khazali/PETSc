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
};
struct _p_DMField {
  PETSCHEADER(struct _DMFieldOps);
  void *data;
  PetscBool setupcalled;
};
#endif
