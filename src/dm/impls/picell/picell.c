#define PETSCDM_DLL
#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/
/* #include <petscdmda.h> */
/* #include <petscsf.h> */
PETSC_EXTERN PetscErrorCode DMPlexView_HDF5(DM, PetscViewer);
#undef __FUNCT__
#define __FUNCT__ "DMView_PICell"
PetscErrorCode DMView_PICell(DM dm, PetscViewer viewer)
{
  PetscBool      iascii, ishdf5, isvtk;
  PetscErrorCode ierr;
  DM_PICell      *dmpi = (DM_PICell *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  if (iascii) {
    ierr = DMView(dmpi->dmplex, viewer);CHKERRQ(ierr);
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ);CHKERRQ(ierr);
    ierr = DMPlexView_HDF5(dmpi->dmplex, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  else if (isvtk) {
    /* ierr = DMPICellVTKWriteAll((PetscObject) dm,viewer);CHKERRQ(ierr); */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_PICell"
PetscErrorCode  DMSetFromOptions_PICell(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscErrorCode ierr;
  DM_PICell      *dmpi = (DM_PICell *) dm->data;
  /* const char *prefix; */

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPICell Options");CHKERRQ(ierr);

  /* ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr); */
  /* ierr = PetscObjectSetOptionsPrefix((PetscObject)dmpi->dmplex,prefix);CHKERRQ(ierr); */
  ierr = DMSetFromOptions(dmpi->dmplex);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_PICell"
PetscErrorCode DMSetUp_PICell(DM dm)
{
  DM_PICell      *dmpi = (DM_PICell *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  /* We have built dmplex, now create vectors */
  ierr = DMSetUp(dmpi->dmplex);CHKERRQ(ierr); /* build a grid */
  ierr = DMCreateGlobalVector(dmpi->dmplex, &dmpi->phi);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->phi, "potential");CHKERRQ(ierr);
  ierr = VecZeroEntries(dmpi->phi);CHKERRQ(ierr);
  ierr = VecDuplicate(dmpi->phi, &dmpi->rho);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->rho, "density");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_PICell"
PetscErrorCode DMDestroy_PICell(DM dm)
{
  DM_PICell      *dmpi = (DM_PICell *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESDestroy(&dmpi->snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dmpi->dmplex);CHKERRQ(ierr);
  ierr = VecDestroy(&dmpi->rho);CHKERRQ(ierr);
  ierr = VecDestroy(&dmpi->phi);CHKERRQ(ierr);
  ierr = PetscFree(dmpi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreate_PICell"
PETSC_EXTERN PetscErrorCode DMCreate_PICell(DM dm)
{
  DM_PICell      *dmpi;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm,&dmpi);CHKERRQ(ierr);

  dm->dim  = 0;
  dm->data = dmpi;

  dm->ops->view                            = DMView_PICell;
  dm->ops->load                            = NULL;
  dm->ops->setfromoptions                  = DMSetFromOptions_PICell;
  dm->ops->clone                           = NULL;
  dm->ops->setup                           = DMSetUp_PICell;
  dm->ops->createdefaultsection            = NULL;
  dm->ops->createdefaultconstraints        = NULL;
  dm->ops->createglobalvector              = NULL;
  dm->ops->createlocalvector               = NULL;
  dm->ops->getlocaltoglobalmapping         = NULL;
  dm->ops->createfieldis                   = NULL;
  dm->ops->createcoordinatedm              = NULL;
  dm->ops->getcoloring                     = NULL;
  dm->ops->creatematrix                    = NULL;
  dm->ops->createinterpolation             = NULL;
  dm->ops->getaggregates                   = NULL;
  dm->ops->getinjection                    = NULL;
  dm->ops->refine                          = NULL;
  dm->ops->coarsen                         = NULL;
  dm->ops->refinehierarchy                 = NULL;
  dm->ops->coarsenhierarchy                = NULL;
  dm->ops->globaltolocalbegin              = NULL;
  dm->ops->globaltolocalend                = NULL;
  dm->ops->localtoglobalbegin              = NULL;
  dm->ops->localtoglobalend                = NULL;
  dm->ops->destroy                         = DMDestroy_PICell;
  dm->ops->createsubdm                     = NULL;
  dm->ops->getdimpoints                    = NULL;
  dm->ops->locatepoints                    = NULL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPICellAddSource"
/* add density 'rho'[1] (vector of size 1) at 'coord'[dim] to global density vector (dmpi->rho) */
PetscErrorCode DMPICellAddSource(DM dm, Vec coord, Vec rho)
{
  DM_PICell      *dmpi = (DM_PICell *) dm->data;
  Vec globalrho = dmpi->rho;
  PetscErrorCode ierr;
  PetscScalar rone=1.;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);

  /* Matt */

  PetscFunctionReturn(0);
}

/* get gradient at point 'coord' and put it in D vector 'jet' */
#undef __FUNCT__
#define __FUNCT__ "DMPICellGetJet"
PetscErrorCode  DMPICellGetJet(DM dm, Vec coord, PetscInt order, Vec jet)
{
  DM_PICell      *dmpi = (DM_PICell *) dm->data;
  Vec globalpot = dmpi->phi;
  PetscErrorCode ierr;
  PetscScalar rone=1.;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);

  /* Matt */
  ierr = VecSet(jet,rone);CHKERRQ(ierr); /* dummy grad now */

  PetscFunctionReturn(0);
}
