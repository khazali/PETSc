#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/

PETSC_INTERN PetscErrorCode DMFieldCreate(DM dm,PetscInt numComponents,DMFieldContinuity continuity,DMField *field)
{
  PetscErrorCode ierr;
  DMField        b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(field,2);
  ierr = DMFieldInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(b,DMFIELD_CLASSID,"DMField","Field over DM","DM",PetscObjectComm((PetscObject)dm),DMFieldDestroy,DMFieldView);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  b->dm = dm;
  b->continuity = continuity;
  b->numComponents = numComponents;
  *field = b;
  PetscFunctionReturn(0);
}

/*@
   DMFieldDestroy - destroy a DMField

   Collective

   Input Arguments:
.  field - address of DMField

   Level: advanced

.seealso: DMFieldCreate()
@*/
PetscErrorCode DMFieldDestroy(DMField *field)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*field) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*field),DMFIELD_CLASSID,1);
  if (--((PetscObject)(*field))->refct > 0) {*field = 0; PetscFunctionReturn(0);}
  if ((*field)->ops->destroy) {ierr = (*(*field)->ops->destroy)(*field);CHKERRQ(ierr);}
  ierr = DMDestroy(&((*field)->dm));CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMFieldView - view a DMField

   Collective

   Input Arguments:
+  field - DMField
-  viewer - viewer to display field, for example PETSC_VIEWER_STDOUT_WORLD

   Level: advanced

.seealso: DMFieldCreate()
@*/
PetscErrorCode DMFieldView(DMField field,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)field),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(field,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)field,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  }
  if (field->ops->view) {ierr = (*field->ops->view)(field,viewer);CHKERRQ(ierr);}
  if (iascii) {
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
