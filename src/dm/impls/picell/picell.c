#define PETSCDM_DLL
#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/
#include <petsc/private/petscfeimpl.h>    /*I   "petscfe.h"   I*/
#include <petsc/private/snesimpl.h>    /*I   "petscsnes.h"   I*/
#include "petscdmforest.h"
#include <petscdmda.h>
#include <petscsf.h>

/* Logging support */
PetscLogEvent DMPICell_Solve, DMPICell_SetUp, DMPICell_AddSource, DMPICell_LocateProcess, DMPICell_GetJet, DMPICell_Add1, DMPICell_Add2, DMPICell_Add3;

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
    ierr = DMView(dmpi->dm, viewer);CHKERRQ(ierr);
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ);CHKERRQ(ierr);
    ierr = DMPlexView_HDF5(dmpi->dm, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dmpi->dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  else if (isvtk) {
    SETERRQ(PetscObjectComm((PetscObject) dmpi->dm), PETSC_ERR_SUP, "VTK not supported in this build");
    /* ierr = DMPICellVTKWriteAll((PetscObject) dm,viewer);CHKERRQ(ierr); */
  }
  else {
    SETERRQ(PetscObjectComm((PetscObject) dmpi->dm), PETSC_ERR_SUP, "Unknown viewer type");
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

  if (!dmpi->dm) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "dm not created");
  ierr = DMSetFromOptions(dmpi->dm);CHKERRQ(ierr);

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

  /* We have built dm, now create vectors */
  if (!dmpi->dm) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "dm not created");
  ierr = DMSetUp(dmpi->dm);CHKERRQ(ierr); /* build a grid */
  ierr = DMCreateGlobalVector(dmpi->dm, &dmpi->phi);CHKERRQ(ierr); /* plex not good yet with p4est */
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
  ierr = DMDestroy(&dmpi->dm);CHKERRQ(ierr);
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
  dmpi->dm = 0;
  dmpi->phi = 0;
  dmpi->rho = 0;
  dmpi->snes = 0;
  dmpi->fem = 0;

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
#define __FUNCT__ "ConvertPlex"
static PetscErrorCode ConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  if (isPlex) {
    *plex = dm;
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex);CHKERRQ(ierr);
    if (!*plex) {
      ierr = DMConvert(dm,DMPLEX,plex);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex);CHKERRQ(ierr);
      if (copy) {
        PetscInt    i;
        PetscObject obj;
        const char *comps[3] = {"A","dmAux","dmCh"};

        ierr = DMCopyDMSNES(dm, *plex);CHKERRQ(ierr);
        for (i = 0; i < 3; i++) {
          ierr = PetscObjectQuery((PetscObject) dm, comps[i], &obj);CHKERRQ(ierr);
          ierr = PetscObjectCompose((PetscObject) *plex, comps[i], obj);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = PetscObjectReference((PetscObject) *plex);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPICellAddSource"
/*@
  DMPICellAddSource - add densities 'densities' at 'coord' to density vector 'rho'

  Input Parameters:
. dm - the DM context
. coord - serial vector coordinates to add at
. densities - serial vector data to add
. cell - cell index in DM

  Output Parameters:
. rho - local vector to add to

  Level: intermediate

.seealso: DMPICellGetJet()
@*/
PetscErrorCode DMPICellAddSource(DM dm, Vec coord, Vec densities, PetscInt cell, Vec rho)
{
  DM_PICell       *dmpi = (DM_PICell *) dm->data;
  Vec             refCoord;
  PetscScalar     *rhoArr, *xx, *xi, *elemVec,sum;
  PetscReal       v0[81], J[243], invJ[243], detJ[27];
  PetscReal       *B = NULL;
  PetscInt        totDim,p,N,dim,b,d,e,order;
  PetscErrorCode  ierr;
  PetscDS         prob;
  DM              plex;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(coord, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(densities, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(rho, VEC_CLASSID, 5);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(DMPICell_AddSource,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPICell_Add1,dm,0,0,0);CHKERRQ(ierr);
#endif
  ierr = VecGetArray(densities, &rhoArr);CHKERRQ(ierr);
  ierr = ConvertPlex(dmpi->dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  order = pow(totDim,1./(double)dim) - 1;
  /* Affine approximation for reference coordinates */
  ierr = DMPlexComputeCellGeometryAffineFEM(plex, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
  /* get coordinates */
  ierr = VecDuplicate(coord, &refCoord);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coord, &b);CHKERRQ(ierr); if(b!=dim) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP, "b = %d", b);
  ierr = VecGetLocalSize(coord, &N);CHKERRQ(ierr);
  if (N%dim) SETERRQ2(PetscObjectComm((PetscObject) plex), PETSC_ERR_PLIB, "N=%D dim=%D",N,dim);
  N /= dim;
  ierr = VecGetArray(coord, &xx);CHKERRQ(ierr);
  ierr = VecGetArray(refCoord, &xi);CHKERRQ(ierr);
  /* apply xi = J^-1 * (x - v0) */
  for (p = 0; p < N; ++p) {
    PetscScalar *pxx = &xx[dim*p], *pxi = &xi[dim*p];
#if defined(__INTEL_COMPILER)
#pragma simd vectorlengthfor(PetscReal)
#endif
    for (d = 0; d < dim; ++d) {
      pxi[d] = -1.;
#if defined(__INTEL_COMPILER)
#pragma simd vectorlengthfor(PetscReal)
#endif
      for (e = 0; e < dim; ++e) {
        pxi[d] += invJ[d*dim+e]*(pxx[e] - v0[e]);
      }
    }
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(DMPICell_Add1,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPICell_Add2,dm,0,0,0);CHKERRQ(ierr);
#endif
  ierr = PetscFEGetTabulation(dmpi->fem, N, xi, &B, NULL, NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(DMPICell_Add2,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPICell_Add3,dm,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMGetWorkArray(plex, totDim, PETSC_SCALAR, &elemVec);CHKERRQ(ierr);
  ierr = PetscMemzero(elemVec, totDim * sizeof(PetscScalar));CHKERRQ(ierr);
#if defined(__INTEL_COMPILER)
  __assume_aligned(B,PETSC_MEMALIGN);
#endif
  for (p = 0; p < N; ++p) {
#if defined(__INTEL_COMPILER)
#pragma simd vectorlengthfor(PetscReal)
#endif
    for (b = 0, sum = 0; b < totDim; ++b) {
      elemVec[b] += B[p*totDim + b] * rhoArr[p];
      sum += B[p*totDim + b];
/* #ifdef PETSC_USE_DEBUG */
      if (order==1 && B[p*totDim + b] < -.1) {
        PetscPrintf(PETSC_COMM_SELF,"\t\tDMPICellAddSource ERROR element %d, p=%d/%d, add B[%d] = %g, x = %12.8e %12.8e %12.8e, ref x = %12.8e %12.8e %12.8e order=%D\n",cell,p+1,N,p*totDim + b,B[p*totDim + b],xx[p*dim+0],xx[p*dim+1],xx[p*dim+2],xi[p*dim+0],xi[p*dim+1],xi[p*dim+2],order);
        /* SETERRQ1(PetscObjectComm((PetscObject) plex), PETSC_ERR_PLIB, "negative interpolant %g, Plex LocatePoint not great with coarse grids",B[p*totDim + b]); */
      }
/* #endif */
    }
    if (PetscAbsReal(1.-sum) > 1.e-7) SETERRQ1(PetscObjectComm((PetscObject) plex), PETSC_ERR_PLIB, "sun interpolant %g",sum);
  }
  ierr = VecRestoreArray(coord, &xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(densities, &rhoArr);CHKERRQ(ierr);
  ierr = DMPlexVecSetClosure(plex, NULL, rho, cell, elemVec, ADD_VALUES);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(plex, totDim, PETSC_SCALAR, &elemVec);CHKERRQ(ierr);
  ierr = PetscFERestoreTabulation(dmpi->fem, N, xi, &B, NULL, NULL);CHKERRQ(ierr);
  ierr = VecRestoreArray(refCoord, &xi);CHKERRQ(ierr);
  ierr = VecDestroy(&refCoord);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(DMPICell_Add3,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPICell_AddSource,dm,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPICellGetJet"
/*@
  DMPICellGetJet - get gradient at point 'coord' and put it in D vector 'jet'

  Input Parameters:
. dm - the DM context
. coord - serial vector coordinates to add at
. philoc - local vector of potentials
. cell - cell index in DM

  Output Parameters:
. jet - vector of gradients at 'coord' points

  Level: intermediate

.seealso: DMPICellAddSource()
@*/
PetscErrorCode  DMPICellGetJet(DM dm, Vec coord, Vec philoc, PetscInt cell, Vec jet)
{
  DM_PICell       *dmpi = (DM_PICell *) dm->data;
  Vec             refCoord;
  PetscScalar     *jetArr, *xx, *xi;
  PetscReal       *D = NULL, *B = NULL;
  PetscReal       v0[81], J[243], invJ[243], detJ[27];
  PetscScalar     *values = NULL;
  PetscInt        totDim,p,N,dim,c,d,e,nloc=27;
  PetscErrorCode  ierr;
  PetscDS         prob;
  DM              plex;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(coord, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(philoc, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(jet, VEC_CLASSID, 5);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(DMPICell_GetJet,dm,0,0,0);CHKERRQ(ierr);
#endif
  ierr = ConvertPlex(dmpi->dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecDuplicate(coord, &refCoord);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coord, &dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coord, &N);CHKERRQ(ierr);
  if (N%dim) SETERRQ2(PetscObjectComm((PetscObject) plex), PETSC_ERR_PLIB, "N=%D dim=%D",N,dim);
  N /= dim;
  ierr = VecGetArray(coord, &xx);CHKERRQ(ierr);
  ierr = VecGetArray(refCoord, &xi);CHKERRQ(ierr);
  /* Affine approximation for reference coordinates */
  ierr = DMPlexComputeCellGeometryAffineFEM(plex, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
  /* apply xi = J^-1 * (x - v0) */
  for (p = 0; p < N; ++p) {
    PetscScalar *pxx = &xx[dim*p], *pxi = &xi[dim*p];
#if defined(__INTEL_COMPILER)
#pragma simd vectorlengthfor(PetscReal)
#endif
    for (d = 0; d < dim; ++d) {
      pxi[d] = -1.;
#if defined(__INTEL_COMPILER)
#pragma simd vectorlengthfor(PetscReal)
#endif
      for (e = 0; e < dim; ++e) {
        pxi[d] += invJ[d*dim+e]*(pxx[e] - v0[e]);
      }
    }
  }
  ierr = VecRestoreArray(coord, &xx);CHKERRQ(ierr);
  ierr = PetscFEGetTabulation(dmpi->fem, N, xi, &B, &D, NULL);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(plex, NULL, philoc, cell, &nloc, &values);CHKERRQ(ierr);
  if (totDim!=nloc) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "totDim!=nloc nloc=%D",nloc);
  if (totDim*9>243) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "totDim*9=%D",totDim*9);
  ierr = VecGetArray(jet, &jetArr);CHKERRQ(ierr);
/* #if defined(PETSC_HAVE_MEMALIGN) */
  /*   __assume_aligned(B,PETSC_MEMALIGN); */
/* #endif */
  /* D[N][totDim][dim] */
  for (p = 0; p < N; ++p) {
    const PetscReal *BB = &B[p*totDim];
    const PetscReal *DD = &D[p*totDim*dim];
    PetscReal *Jet = &jetArr[p*dim];
#if 1
    for (e = 0; e < dim; ++e) {
#if defined(__INTEL_COMPILER)
#pragma simd vectorlengthfor(PetscReal)
#endif
      for (c = 0, Jet[e] = 0; c < totDim; ++c) {
        Jet[e] += BB[c] * DD[c*dim+e] * values[c];
      }
    }
#else
    for (e = 0; e < dim; ++e) Jet[e] = 0;
    for (c = 0; c < totDim; ++c) {
      const PetscReal *DDD = &DD[c*dim];
      for (e = 0; e < dim; ++e) Jet[e] += BB[c] * DDD[e] * values[c];
    }
#endif
  }
  ierr = DMPlexVecRestoreClosure(plex, NULL, philoc, cell, &nloc, &values);CHKERRQ(ierr);
  ierr = VecRestoreArray(jet, &jetArr);CHKERRQ(ierr);
  ierr = PetscFERestoreTabulation(dmpi->fem, N, xi, &B, &D, NULL);CHKERRQ(ierr);
  ierr = VecRestoreArray(refCoord, &xi);CHKERRQ(ierr);
  ierr = VecDestroy(&refCoord);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(DMPICell_GetJet,dm,0,0,0);CHKERRQ(ierr);
#endif
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
