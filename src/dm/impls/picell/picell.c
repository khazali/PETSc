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
    SETERRQ(PetscObjectComm((PetscObject) dmpi->dmplex), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  else if (isvtk) {
    SETERRQ(PetscObjectComm((PetscObject) dmpi->dmplex), PETSC_ERR_SUP, "VTK not supported in this build");
    /* ierr = DMPICellVTKWriteAll((PetscObject) dm,viewer);CHKERRQ(ierr); */
  }
  else {
    SETERRQ(PetscObjectComm((PetscObject) dmpi->dmplex), PETSC_ERR_SUP, "Unknown viewer type");
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

  if (!dmpi->dmplex) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "dmplex not created");
  ierr = DMSetFromOptions(dmpi->dmplex);CHKERRQ(ierr);
  if (dmpi->dmgrid && dmpi->dmplex != dmpi->dmgrid) {
    ierr = DMSetFromOptions(dmpi->dmgrid);CHKERRQ(ierr);
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

  /* We have built dmplex, now create vectors */
  if (!dmpi->dmplex) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "dmplex not created");
  ierr = DMSetUp(dmpi->dmplex);CHKERRQ(ierr); /* build a grid */
  if (dmpi->dmgrid && dmpi->dmplex != dmpi->dmgrid) {
    ierr = DMSetUp(dmpi->dmgrid);CHKERRQ(ierr);
  }
  ierr = DMCreateGlobalVector(dmpi->dmplex, &dmpi->phi);CHKERRQ(ierr); /* plex not good yet with p4est */
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
  if (dmpi->dmgrid && dmpi->dmplex != dmpi->dmgrid) {
    ierr = DMDestroy(&dmpi->dmgrid);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dmpi->dmplex);CHKERRQ(ierr);
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
/* add densities 'densities' at 'coord' to density vector 'rho' */
/* use dmplex here and not dmgrid (p4est) */
PetscErrorCode DMPICellAddSource(DM dm, Vec coord, Vec densities, PetscInt cell, Vec rho)
{
  DM_PICell    *dmpi = (DM_PICell *) dm->data;
  Vec          refCoord;
  PetscScalar  *lrho, *xx, *xi, *elemVec, *piJ;
  PetscReal    *B = NULL;
  PetscReal    v0[81], J[243], invJ[243], detJ[27], J0inv[9],v00[3], d3 = 8;
  const PetscReal *weights;
  PetscInt     totDim,p,N,dim,b,Nq,qdim,q,d,e;
  PetscErrorCode ierr;
  PetscDS        prob;
  PetscQuadrature  quad;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(coord, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(densities, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(rho, VEC_CLASSID, 5);
  ierr = PetscLogEventBegin(DMPICell_AddSource,dm,0,0,0);CHKERRQ(ierr);
  ierr = VecDuplicate(coord, &refCoord);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coord, &dim);CHKERRQ(ierr);
  d3 = pow(2,dim);
  ierr = VecGetLocalSize(coord, &N);CHKERRQ(ierr);
  if (N%dim) SETERRQ2(PetscObjectComm((PetscObject) dmpi->dmplex), PETSC_ERR_PLIB, "N=%D dim=%D",N,dim);
  N /= dim;
  ierr = VecGetArray(coord, &xx);CHKERRQ(ierr);
  ierr = VecGetArray(refCoord, &xi);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(dmpi->fem, &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, &qdim, &Nq, NULL, &weights);CHKERRQ(ierr);
  if (qdim != dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Point dimension %d != quadrature dimension %d", dim, qdim);
  if (Nq > 27) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nq > 27", Nq);
  /* Affine approximation for reference coordinates */
  ierr = DMPlexComputeCellGeometryFEM(dmpi->dmplex, cell, dmpi->fem, v0, J, invJ, detJ);CHKERRQ(ierr);
  /* get average for now */
  for (e = 0; e < dim; ++e) v00[e] = 0;
  for (e = 0; e < dim*dim; ++e) J0inv[e] = 0;
  for (q = 0, piJ = &invJ[0]; q < Nq; ++q, piJ += dim*dim) {
    for (e = 0; e < dim; ++e) {
      v00[e] += weights[q]*v0[q*dim + e];
      for (d = 0; d < dim; ++d) {
        J0inv[e*dim + d] += weights[q]*piJ[e*dim + d];
      }
    }
  }
  for (e = 0; e < dim; ++e) v00[e] /= d3;
  for (e = 0; e < dim*dim; ++e) J0inv[e] /= d3;
  /* apply xi = J^-1 * (x - v0) */
  for (p = 0; p < N; ++p) {
    PetscScalar *pxx = &xx[dim*p], *pxi = &xi[dim*p];
    for (e = 0; e < dim; ++e) {
      pxi[e] = 0;
      for (d = 0; d < dim; ++d) {
        pxi[e] += J0inv[e*dim + d]*(pxx[d] - v00[d]);
      }
    }
  }
  ierr = VecRestoreArray(coord, &xx);CHKERRQ(ierr);
  ierr = PetscFEGetTabulation(dmpi->fem, N, xi, &B, NULL, NULL);CHKERRQ(ierr);
  ierr = VecRestoreArray(refCoord, &xi);CHKERRQ(ierr);
  ierr = VecDestroy(&refCoord);CHKERRQ(ierr);
  ierr = DMGetDS(dmpi->dmplex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dmpi->dmplex, totDim, PETSC_SCALAR, &elemVec);CHKERRQ(ierr);
  ierr = PetscMemzero(elemVec, totDim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecGetArray(densities, &lrho);CHKERRQ(ierr);
  for (p = 0; p < N; ++p) {
    for (b = 0; b < totDim; ++b) {
      elemVec[b] += B[p*totDim + b] * lrho[p];
#ifdef PETSC_USE_DEBUG
      if (B[p*totDim + b] < -.1) {
        PetscPrintf(PETSC_COMM_SELF,"DMPICellAddSource ERROR (Plex LocatePoint not great with coarse grids) elem %d, p=%d/%d, add B[%d]. B = %g %g %g %g %g %g %g %g ... \n",cell,p,N,p*totDim + b,B[p*totDim + b],B[0],B[1],B[2],B[3],B[4],B[5],B[6],B[7]);
        SETERRQ1(PetscObjectComm((PetscObject) dmpi->dmplex), PETSC_ERR_PLIB, "negative interpolant %g, Plex LocatePoint not great with coarse grids",B[b*N + p]);
      }
#endif
    }
  }
  ierr = VecRestoreArray(densities, &lrho);CHKERRQ(ierr);
  ierr = DMPlexVecSetClosure(dmpi->dmplex, NULL, rho, cell, elemVec, ADD_VALUES);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dmpi->dmplex, totDim, PETSC_SCALAR, &elemVec);CHKERRQ(ierr);
  ierr = PetscFERestoreTabulation(dmpi->fem, N, xi, &B, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPICell_AddSource,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* get gradient at point 'coord' and put it in D vector 'jet' */
#undef __FUNCT__
#define __FUNCT__ "DMPICellGetJet"
PetscErrorCode  DMPICellGetJet(DM dm, Vec coord, PetscInt order, Vec jet, PetscInt cell)
{
  /* DM_PICell      *dmpi = (DM_PICell *) dm->data; */
  PetscErrorCode ierr;
  PetscScalar rone=1.;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(coord, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(jet, VEC_CLASSID, 4);
  /* ierr = PetscLogEventBegin(DMPICell_GetJet,a_dm,0,0,0);CHKERRQ(ierr); */

  /* Matt */
  ierr = VecSet(jet,rone);CHKERRQ(ierr); /* dummy grad now */

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
