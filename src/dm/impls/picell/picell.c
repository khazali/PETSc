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
  DM_PICell      *mesh = (DM_PICell *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  if (iascii) {
    /* ierr = DMPICellView_Ascii(dm, viewer);CHKERRQ(ierr); */
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ);CHKERRQ(ierr);
    ierr = DMPlexView_HDF5(mesh->dmplex, viewer);CHKERRQ(ierr);
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
PetscErrorCode  DMSetFromOptions_PICell(PetscOptions *PetscOptionsObject,DM dm)
{
  PetscErrorCode ierr;
  DM_PICell      *mesh = (DM_PICell *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPICell Options");CHKERRQ(ierr);
  
  /* Handle DMPICell refinement */
  /* ierr = PetscOptionsInt("-dm_refine", "The number of uniform refinements", "DMCreate", refine, &refine, NULL);CHKERRQ(ierr); */

  ierr = DMSetFromOptions(mesh->dmplex);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_PICell"
PetscErrorCode DMSetUp_PICell(DM dm)
{
  DM_PICell      *mesh = (DM_PICell *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);

  /* We have built dmplex, now create vectors */
  ierr = DMCreateGlobalVector(mesh->dmplex, &mesh->phi);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) mesh->phi, "potential");CHKERRQ(ierr);
  ierr = VecDuplicate(mesh->phi, &mesh->rho);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) mesh->rho, "density");CHKERRQ(ierr);

  /* set up solver */
  {
    MPI_Comm       comm;
    ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
    ierr = SNESCreate(comm,&mesh->snes);CHKERRQ(ierr);
    ierr = SNESSetDM(mesh->snes,dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_PICell"
PetscErrorCode DMDestroy_PICell(DM dm)
{
  DM_PICell      *mesh = (DM_PICell *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESDestroy(&mesh->snes);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh->dmplex);CHKERRQ(ierr);
  ierr = VecDestroy(&mesh->rho);CHKERRQ(ierr);
  ierr = VecDestroy(&mesh->phi);CHKERRQ(ierr);
  ierr = PetscFree(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreate_PICell"
PETSC_EXTERN PetscErrorCode DMCreate_PICell(DM dm)
{
  DM_PICell      *mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm,&mesh);CHKERRQ(ierr);

  dm->dim  = 0;
  dm->data = mesh;

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
#define __FUNCT__ "DMPICellAddRho"
PetscErrorCode DMPICellAddRho(DM dm, double r, double z, double phi, PetscReal val)
{
  DM_PICell      *mesh = (DM_PICell *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);



  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPICellGetGrad"
PetscErrorCode DMPICellGetGrad(DM dm, double r, double z, double phi, double gradphi[])
{
  DM_PICell      *mesh = (DM_PICell *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);



  
  PetscFunctionReturn(0);
}
