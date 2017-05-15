#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/

typedef struct _n_DMField_DA
{
  PetscScalar *cornerVals;
}
DMField_DA;

static PetscErrorCode DMFieldDestroy_DA(DMField field)
{
  DMField_DA     *dafield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dafield = (DMField_DA *) field->data;
  ierr = PetscFree(dafield->cornerValues);CHKERRQ(ierr);
  ierr = PetscFree(dafield);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldInitialize_DA(DMField field)
{
  PetscFunctionBegin;
  field->ops->destroy = DMFieldDestroy_DA;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMFieldCreate_DA(DMField field)
{
  DMField_DA     *dafield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(field,&dafield);CHKERRQ(ierr);
  field->data = dafield;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldCreateDA(DM dm, PetscInt numComponents, const PetscScalar *cornerValues,DMField *field)
{
  DMField        b;
  DMField_DA     *dafield;
  PetscInt       dim, nv, i;
  PetscScalar    *cv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMFieldCreate(dm,numComponents,DMFIELD_VERTEX,&b);CHKERRQ(ierr);
  dafield = (DMField_DA *) b->data;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  nv = (1 << dim) * numComponents;
  ierr = PetscMalloc1(nv,&cv);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) cv[i] = cornerValues[i];
  dafield->cornerVals = cv;
  *field = b;
  PetscFunctionReturn(0);
}
