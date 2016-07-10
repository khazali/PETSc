#define PETSCDM_DLL
#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/
#include <petsc/private/petscfeimpl.h>    /*I   "petscfe.h"   I*/
#include "petscdmforest.h"
#include <petscdmda.h>
#include <petscsf.h>

/* Logging support */
PetscLogEvent DMPICell_Solve, DMPICell_SetUp, DMPICell_AddSource, DMPICell_LocateProcess, DMPICell_GetJet;

PETSC_EXTERN PetscErrorCode DMPlexView_HDF5(DM, PetscViewer);
#undef __FUNCT__
#define __FUNCT__ "DMView_PICell"
PetscErrorCode DMView_PICell(DM a_dm, PetscViewer viewer)
{
  PetscBool      iascii, ishdf5, isvtk;
  PetscErrorCode ierr;
  DM_PICell      *dmpi = (DM_PICell *) a_dm->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(a_dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  if (iascii) {
    ierr = DMView(dmpi->dmgrid, viewer);CHKERRQ(ierr);
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ);CHKERRQ(ierr);
    ierr = DMPlexView_HDF5(dmpi->dmplex, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dmpi->dmgrid), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  else if (isvtk) {
    SETERRQ(PetscObjectComm((PetscObject) dmpi->dmgrid), PETSC_ERR_SUP, "VTK not supported in this build");
    /* ierr = DMPICellVTKWriteAll((PetscObject) dm,viewer);CHKERRQ(ierr); */
  }
  else {
    SETERRQ(PetscObjectComm((PetscObject) dmpi->dmgrid), PETSC_ERR_SUP, "Unknown viewer type");
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
  /* ierr = PetscObjectSetOptionsPrefix((PetscObject)dmpi->dmgrid,prefix);CHKERRQ(ierr); */
  if (!dmpi->dmgrid) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "dmgrid not created");
  ierr = DMSetFromOptions(dmpi->dmgrid);CHKERRQ(ierr);
  if (dmpi->dmplex && dmpi->dmplex != dmpi->dmgrid) {
    ierr = DMSetFromOptions(dmpi->dmplex);CHKERRQ(ierr);
  }

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
  ierr = PetscLogEventBegin(DMPICell_SetUp,dm,0,0,0);CHKERRQ(ierr);

  /* We have built dmgrid, now create vectors */
  if (!dmpi->dmgrid) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "dmgrid not created");
  ierr = DMSetUp(dmpi->dmgrid);CHKERRQ(ierr); /* build a grid */
  if (dmpi->dmplex && dmpi->dmplex != dmpi->dmgrid) {
    ierr = DMSetUp(dmpi->dmplex);CHKERRQ(ierr);
  }
  ierr = DMCreateGlobalVector(dmpi->dmgrid, &dmpi->phi);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->phi, "phi");CHKERRQ(ierr);
  ierr = VecZeroEntries(dmpi->phi);CHKERRQ(ierr);
  ierr = VecDuplicate(dmpi->phi, &dmpi->rho);CHKERRQ(ierr);
  ierr = VecZeroEntries(dmpi->rho);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->rho, "rho");CHKERRQ(ierr);

  ierr = PetscLogEventEnd(DMPICell_SetUp,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_PICell"
PetscErrorCode DMDestroy_PICell(DM dm)
{
  DM_PICell      *dmpi = (DM_PICell *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&dmpi->rho);CHKERRQ(ierr);
  ierr = VecDestroy(&dmpi->phi);CHKERRQ(ierr);
  if (dmpi->dmplex != dmpi->dmgrid) {
    ierr = DMDestroy(&dmpi->dmplex);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dmpi->dmgrid);CHKERRQ(ierr);
  ierr = SNESDestroy(&dmpi->snes);CHKERRQ(ierr);
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
  dmpi->dmgrid = 0;
  dmpi->dmplex = 0;
  dmpi->phi = 0;
  dmpi->rho = 0;
  dmpi->snes = 0;

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
/* add densities 'src' at 'coord' to local density vector 'locrho' */
/* use dmplex here and not dmgrid (p4est) */
PetscErrorCode DMPICellAddSource(DM dm, Vec coord, Vec src, PetscInt cell, Vec locrho)
{
  DM_PICell    *dmpi = (DM_PICell *) dm->data;
  Vec          refCoord;
  PetscScalar  *x, *xi, *elemVec;
  PetscReal    *B = NULL;
  PetscReal    v0[3], J[9], invJ[9], detJ;
  PetscInt     totDim,p,N,dim,b;
  PetscErrorCode ierr;
  PetscDS        prob;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(coord, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(src, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(locrho, VEC_CLASSID, 5);
  ierr = PetscLogEventBegin(DMPICell_AddSource,dm,0,0,0);CHKERRQ(ierr);
  ierr = VecDuplicate(coord, &refCoord);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coord, &dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coord, &N);CHKERRQ(ierr);
  if (N%dim) SETERRQ2(PetscObjectComm((PetscObject) dmpi->dmplex), PETSC_ERR_SUP, "N=%D dim=%D",N,dim);
  N   /= dim;
  ierr = VecGetArray(coord, &x);CHKERRQ(ierr);
  ierr = VecGetArray(refCoord, &xi);CHKERRQ(ierr);
  /* Affine approximation for reference coordinates */
  ierr = DMPlexComputeCellGeometryFEM(dmpi->dmplex, cell, dmpi->fem, v0, J, invJ, &detJ);CHKERRQ(ierr);
  for (p = 0; p < N; ++p) {
    CoordinatesRealToRef(dim, dim, v0, invJ, &x[p*dim], &xi[p*dim]);
  }
  ierr = VecRestoreArray(coord, &x);CHKERRQ(ierr);
  ierr = PetscFEGetTabulation(dmpi->fem, N, xi, &B, NULL, NULL);CHKERRQ(ierr);
  ierr = VecRestoreArray(refCoord, &xi);CHKERRQ(ierr);
  ierr = VecDestroy(&refCoord);CHKERRQ(ierr);
  ierr = DMGetDS(dmpi->dmplex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dmpi->dmplex, totDim, PETSC_SCALAR, &elemVec);CHKERRQ(ierr);
  ierr = PetscMemzero(elemVec, totDim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecGetArray(src, &x);CHKERRQ(ierr);
  for (b = 0; b < totDim; ++b) {
    for (p = 0; p < N; ++p) {
      elemVec[b] += B[b*N + p] * x[p];
    }
  }
  ierr = VecRestoreArray(src, &x);CHKERRQ(ierr);
  /* PetscPrintf(PETSC_COMM_SELF,"\t\t\t\tDMPICellAddSource elem %d, add v=(%g,%g,%g,%g,%g,%g,%g,%g)\n",cell,elemVec[0],elemVec[1],elemVec[2],elemVec[3],elemVec[4],elemVec[5],elemVec[6],elemVec[7]); */
  ierr = DMPlexVecSetClosure(dmpi->dmplex, NULL, locrho, cell, elemVec, ADD_VALUES);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dmpi->dmplex, totDim, PETSC_SCALAR, &elemVec);CHKERRQ(ierr);
  ierr = PetscFERestoreTabulation(dmpi->fem, N, xi, &B, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPICell_AddSource,dm,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* get gradient at point 'coord' and put it in D vector 'jet' */
#undef __FUNCT__
#define __FUNCT__ "DMPICellGetJet"
PetscErrorCode  DMPICellGetJet(DM a_dm, Vec coord, PetscInt order, Vec jet, PetscInt cell)
{
  DM_PICell      *dmpi = (DM_PICell *) a_dm->data;
  PetscErrorCode ierr;
  PetscScalar rone=1.;
  DM dm;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(a_dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(coord, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(jet, VEC_CLASSID, 4);
  /* ierr = PetscLogEventBegin(DMPICell_GetJet,a_dm,0,0,0);CHKERRQ(ierr); */

  ierr = VecSet(jet,rone);CHKERRQ(ierr); /* dummy grad now */

  /* Matt */
  dm = dmpi->dmgrid; /* the DM for the mesh (not really a plex!) */

  /* ierr = PetscLogEventEnd(DMPICell_GetJet,a_dm,0,0,0);CHKERRQ(ierr); */

  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DMGetCellChart"
PetscErrorCode DMGetCellChart(DM dm, PetscInt *cStart, PetscInt *cEnd)
{
  PetscErrorCode ierr;
  PetscBool isForest;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMIsForest(dm,&isForest);CHKERRQ(ierr);
  if (isForest) {
    ierr = DMForestGetCellChart(dm, cStart, cEnd);CHKERRQ(ierr);
  }
  else {
    ierr = DMPlexGetHeightStratum(dm, 0, cStart, cEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
