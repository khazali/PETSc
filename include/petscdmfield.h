#if !defined(__PETSCDMFIELD_H)
#define      __PETSCDMFIELD_H
#include <petscdm.h>
#include <petscdt.h>

/*S
    DMField - PETSc object for defining a field on a mesh topology

    Level: intermediate
S*/
typedef struct _p_DMField* DMField;

PETSC_EXTERN PetscErrorCode DMFieldInitializePackage(void);

PETSC_EXTERN PetscClassId DMFIELD_CLASSID;

/*J
    DMFieldType - String with the name of a DMField method

    Level: intermediate
J*/
typedef const char *DMFieldType;
#define DMFIELDDA       "da"
#define DMFIELDMAPPED   "mapped"
#define DMFIELDSHELL    "shell"

PETSC_EXTERN PetscFunctionList DMFieldList;
PETSC_EXTERN PetscErrorCode    DMFieldSetType(DMField, DMFieldType);
PETSC_EXTERN PetscErrorCode    DMFieldGetType(DMField, DMFieldType*);
PETSC_EXTERN PetscErrorCode    DMFieldRegister(const char[],PetscErrorCode (*)(DMField));

typedef enum {DMFIELD_VERTEX,DMFIELD_EDGE,DMFIELD_FACET,DMFIELD_CELL} DMFieldContinuity;
typedef enum {DMFIELD_SIMPLE,DMFIELD_COVARIANT,DMFIELD_CONTRAVARIANT} DMFieldPullback;

PETSC_EXTERN PetscErrorCode    DMFieldCreate(DM,DMField*);
PETSC_EXTERN PetscErrorCode    DMFieldSetNumComponents(DMField,PetscInt);
PETSC_EXTERN PetscErrorCode    DMFieldGetNumComponents(DMField,PetscInt*);
PETSC_EXTERN PetscErrorCode    DMFieldSetContinuity(DMField,DMFieldContinuity);
PETSC_EXTERN PetscErrorCode    DMFieldGetContinuity(DMField,DMFieldContinuity*);
PETSC_EXTERN PetscErrorCode    DMFieldSetPullback(DMField,DMFieldPullback);
PETSC_EXTERN PetscErrorCode    DMFieldGetPullback(DMField,DMFieldPullback*);
PETSC_EXTERN PetscErrorCode    DMFieldSetField(DMField,PetscDatatype);
PETSC_EXTERN PetscErrorCode    DMFieldGetField(DMField,PetscDatatype*);

PETSC_EXTERN PetscErrorCode    DMFieldSetFromOptions(DMField);
PETSC_EXTERN PetscErrorCode    DMFieldSetUp(DMField);

PETSC_EXTERN PetscErrorCode    DMFieldEvaluate(DMField,PetscInt,PetscQuadrature,PetscScalar*,PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode    DMFieldEvaluateReal(DMField,PetscInt,PetscQuadrature,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode    DMFieldEvaluateFE(DMField,PetscInt,const PetscInt*,PetscQuadrature,PetscScalar*,PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode    DMFieldEvaluateFEReal(DMField,PetscInt,const PetscInt*,PetscQuadrature,PetscReal*,PetscReal*,PetscReal*);

#endif
