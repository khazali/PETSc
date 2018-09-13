#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>
#include <petsc/private/hashmapij.h>
#include <petscao.h>

/* HashIJKL */

#include <petsc/private/hashmap.h>

typedef struct _PetscHashIJKLKey { PetscInt i, j, k, l; } PetscHashIJKLKey;

#define PetscHashIJKLKeyHash(key) \
  PetscHashCombine(PetscHashCombine(PetscHashInt((key).i),PetscHashInt((key).j)), \
                   PetscHashCombine(PetscHashInt((key).k),PetscHashInt((key).l)))

#define PetscHashIJKLKeyEqual(k1,k2) \
  (((k1).i==(k2).i) ? ((k1).j==(k2).j) ? ((k1).k==(k2).k) ? ((k1).l==(k2).l) : 0 : 0 : 0)

PETSC_HASH_MAP(HashIJKL, PetscHashIJKLKey, PetscInt, PetscHashIJKLKeyHash, PetscHashIJKLKeyEqual, -1)


/*
  DMPlexGetFaces_Internal - Gets groups of vertices that correspond to faces for the given cell
  This assumes that the mesh is not interpolated from the depth of point p to the vertices
*/
PetscErrorCode DMPlexGetFaces_Internal(DM dm, PetscInt dim, PetscInt p, PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  const PetscInt *cone = NULL;
  PetscInt        coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetRawFaces_Internal(dm, dim, coneSize, cone, numFaces, faceSize, faces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexRestoreFaces_Internal - Restores the array
*/
PetscErrorCode DMPlexRestoreFaces_Internal(DM dm, PetscInt dim, PetscInt p, PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (faces) { ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, (void *) faces);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*
  DMPlexGetRawFaces_Internal - Gets groups of vertices that correspond to faces for the given cone
*/
PetscErrorCode DMPlexGetRawFaces_Internal(DM dm, PetscInt dim, PetscInt coneSize, const PetscInt cone[], PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  PetscInt       *facesTmp;
  PetscInt        maxConeSize, maxSupportSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (faces && coneSize) PetscValidIntPointer(cone,4);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  if (faces) {ierr = DMGetWorkArray(dm, PetscSqr(PetscMax(maxConeSize, maxSupportSize)), MPIU_INT, &facesTmp);CHKERRQ(ierr);}
  switch (dim) {
  case 1:
    switch (coneSize) {
    case 2:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces = 2;
      if (faceSize) *faceSize = 1;
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  case 2:
    switch (coneSize) {
    case 3:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[0];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces = 3;
      if (faceSize) *faceSize = 2;
      break;
    case 4:
      /* Vertices follow right hand rule */
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[3];
        facesTmp[6] = cone[3]; facesTmp[7] = cone[0];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces = 4;
      if (faceSize) *faceSize = 2;
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  case 3:
    switch (coneSize) {
    case 3:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[0];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces = 3;
      if (faceSize) *faceSize = 2;
      break;
    case 4:
      /* Vertices of first face follow right hand rule and normal points away from last vertex */
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2];
        facesTmp[3] = cone[0]; facesTmp[4]  = cone[3]; facesTmp[5]  = cone[1];
        facesTmp[6] = cone[0]; facesTmp[7]  = cone[2]; facesTmp[8]  = cone[3];
        facesTmp[9] = cone[2]; facesTmp[10] = cone[1]; facesTmp[11] = cone[3];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces = 4;
      if (faceSize) *faceSize = 3;
      break;
    case 8:
      /*  7--------6
         /|       /|
        / |      / |
       4--------5  |
       |  |     |  |
       |  |     |  |
       |  1--------2
       | /      | /
       |/       |/
       0--------3
       */
      if (faces) {
        facesTmp[0]  = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2]; facesTmp[3]  = cone[3]; /* Bottom */
        facesTmp[4]  = cone[4]; facesTmp[5]  = cone[5]; facesTmp[6]  = cone[6]; facesTmp[7]  = cone[7]; /* Top */
        facesTmp[8]  = cone[0]; facesTmp[9]  = cone[3]; facesTmp[10] = cone[5]; facesTmp[11] = cone[4]; /* Front */
        facesTmp[12] = cone[2]; facesTmp[13] = cone[1]; facesTmp[14] = cone[7]; facesTmp[15] = cone[6]; /* Back */
        facesTmp[16] = cone[3]; facesTmp[17] = cone[2]; facesTmp[18] = cone[6]; facesTmp[19] = cone[5]; /* Right */
        facesTmp[20] = cone[0]; facesTmp[21] = cone[4]; facesTmp[22] = cone[7]; facesTmp[23] = cone[1]; /* Left */
        *faces = facesTmp;
      }
      if (numFaces) *numFaces = 6;
      if (faceSize) *faceSize = 4;
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not supported", dim);
  }
  PetscFunctionReturn(0);
}

/*
  DMPlexGetRawFacesHybrid_Internal - Gets groups of vertices that correspond to faces for the given cone using hybrid ordering (prisms)
*/
static PetscErrorCode DMPlexGetRawFacesHybrid_Internal(DM dm, PetscInt dim, PetscInt coneSize, const PetscInt cone[], PetscInt *numFaces, PetscInt *numFacesNotH, PetscInt *faceSize, const PetscInt *faces[])
{
  PetscInt       *facesTmp;
  PetscInt        maxConeSize, maxSupportSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (faces && coneSize) PetscValidIntPointer(cone,4);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  if (faces) {ierr = DMGetWorkArray(dm, PetscSqr(PetscMax(maxConeSize, maxSupportSize)), MPIU_INT, &facesTmp);CHKERRQ(ierr);}
  switch (dim) {
  case 1:
    switch (coneSize) {
    case 2:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        *faces = facesTmp;
      }
      if (numFaces)     *numFaces = 2;
      if (numFacesNotH) *numFacesNotH = 2;
      if (faceSize)     *faceSize = 1;
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  case 2:
    switch (coneSize) {
    case 4:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[2]; facesTmp[3] = cone[3];
        facesTmp[4] = cone[0]; facesTmp[5] = cone[2];
        facesTmp[6] = cone[1]; facesTmp[7] = cone[3];
        *faces = facesTmp;
      }
      if (numFaces)     *numFaces = 4;
      if (numFacesNotH) *numFacesNotH = 2;
      if (faceSize)     *faceSize = 2;
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  case 3:
    switch (coneSize) {
    case 6: /* triangular prism */
      if (faces) {
        facesTmp[0]  = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2]; facesTmp[3]  = -1;      /* Bottom */
        facesTmp[4]  = cone[3]; facesTmp[5]  = cone[4]; facesTmp[6]  = cone[5]; facesTmp[7]  = -1;      /* Top */
        facesTmp[8]  = cone[0]; facesTmp[9]  = cone[1]; facesTmp[10] = cone[3]; facesTmp[11] = cone[4]; /* Back left */
        facesTmp[12] = cone[1]; facesTmp[13] = cone[2]; facesTmp[14] = cone[4]; facesTmp[15] = cone[5]; /* Back right */
        facesTmp[16] = cone[2]; facesTmp[17] = cone[0]; facesTmp[18] = cone[5]; facesTmp[19] = cone[3]; /* Front */
        *faces = facesTmp;
      }
      if (numFaces)     *numFaces = 5;
      if (numFacesNotH) *numFacesNotH = 2;
      if (faceSize)     *faceSize = -4;
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not supported", dim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexRestoreRawFacesHybrid_Internal(DM dm, PetscInt dim, PetscInt coneSize, const PetscInt cone[], PetscInt *numFaces, PetscInt *numFacesNotH, PetscInt *faceSize, const PetscInt *faces[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (faces) { ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, (void *) faces);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetFacesHybrid_Internal(DM dm, PetscInt dim, PetscInt p, PetscInt *numFaces, PetscInt *numFacesNotH, PetscInt *faceSize, const PetscInt *faces[])
{
  const PetscInt *cone = NULL;
  PetscInt        coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetRawFacesHybrid_Internal(dm, dim, coneSize, cone, numFaces, numFacesNotH, faceSize, faces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AreAllConePointsInArray_Private(DM dm, PetscInt p, PetscInt npoints, const PetscInt *points, PetscBool *flg)
{
  PetscInt i,l,n;
  const PetscInt *cone;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  ierr = DMPlexGetConeSize(dm, p, &n);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscFindInt(cone[i], npoints, points, &l);CHKERRQ(ierr);
    if (l < 0) {
      *flg = PETSC_FALSE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/* TODO this is hotfix only that should be replace by actual fix */
static PetscErrorCode DMPlexHotfixInterpolatedPointSF_Private(DM dm)
{
  PetscSF sf;
  PetscInt i,p;
  PetscInt nroots,nleaves,nleaves1;
  const PetscInt *ilocal;
  const PetscSFNode *iremote;
  PetscInt *ilocal1;
  PetscSFNode *iremote1;
  PetscBool flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote);CHKERRQ(ierr);
  ierr = PetscMalloc2(nleaves, &ilocal1, nleaves, &iremote1);CHKERRQ(ierr);
  ierr = PetscMemcpy(ilocal1, ilocal, nleaves*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(iremote1, iremote, nleaves*sizeof(PetscSFNode));CHKERRQ(ierr);
  nleaves1 = nleaves;

  /* 1) if some point p is in interface, then all its cone points must be also in interface  */
  for (i=0; i<nleaves; i++) {
    p = ilocal1[i];
    ierr = AreAllConePointsInArray_Private(dm, p, nleaves1, ilocal1, &flg);CHKERRQ(ierr);
    if (!flg) {
      /* remove p - shift all following points one position back */
      ierr = PetscMemmove(&ilocal1[i], &ilocal1[i+1], (nleaves1-(i+1))*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemmove(&iremote1[i], &iremote1[i+1], (nleaves1-(i+1))*sizeof(PetscSFNode));CHKERRQ(ierr);
      nleaves1--;
      i--;
    }
  }

  ierr = PetscSFSetGraph(sf, nroots, nleaves1, ilocal1, PETSC_COPY_VALUES, iremote1, PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)sf,NULL,"-fixed_sf_view");CHKERRQ(ierr);
  ierr = PetscFree2(ilocal1, iremote1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This interpolates faces for cells at some stratum */
static PetscErrorCode DMPlexInterpolateFaces_Internal(DM dm, PetscInt cellDepth, DM idm)
{
  DMLabel        subpointMap;
  PetscHashIJKL  faceTable;
  PetscInt      *pStart, *pEnd;
  PetscInt       cellDim, depth, faceDepth = cellDepth, numPoints = 0, faceSizeAll = 0, face, c, d;
  PetscInt       coneSizeH = 0, faceSizeAllH = 0, numCellFacesH = 0, faceH, pMax = -1, dim, outerloop;
  PetscInt       cMax, fMax, eMax, vMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &cellDim);CHKERRQ(ierr);
  /* HACK: I need a better way to determine face dimension, or an alternative to GetFaces() */
  ierr = DMPlexGetSubpointMap(dm, &subpointMap);CHKERRQ(ierr);
  if (subpointMap) ++cellDim;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ++depth;
  ++cellDepth;
  cellDim -= depth - cellDepth;
  ierr = PetscMalloc2(depth+1,&pStart,depth+1,&pEnd);CHKERRQ(ierr);
  for (d = depth-1; d >= faceDepth; --d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d+1], &pEnd[d+1]);CHKERRQ(ierr);
  }
  ierr = DMPlexGetDepthStratum(dm, -1, NULL, &pStart[faceDepth]);CHKERRQ(ierr);
  pEnd[faceDepth] = pStart[faceDepth];
  for (d = faceDepth-1; d >= 0; --d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
  }
  cMax = fMax = eMax = vMax = PETSC_DETERMINE;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (cellDim == dim) {
    ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
    pMax = cMax;
  } else if (cellDim == dim -1) {
    ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, NULL, NULL);CHKERRQ(ierr);
    pMax = fMax;
  }
  pMax = pMax < 0 ? pEnd[cellDepth] : pMax;
  if (pMax < pEnd[cellDepth]) {
    const PetscInt *cellFaces, *cone;
    PetscInt        numCellFacesT, faceSize, cf;

    ierr = DMPlexGetConeSize(dm, pMax, &coneSizeH);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, pMax, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetRawFacesHybrid_Internal(dm, cellDim, coneSizeH, cone, &numCellFacesH, &numCellFacesT, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize < 0) {
      PetscInt *sizes, minv, maxv;

      /* count vertices of hybrid and non-hybrid faces */
      ierr = PetscCalloc1(numCellFacesH, &sizes);CHKERRQ(ierr);
      for (cf = 0; cf < numCellFacesT; ++cf) { /* These are the non-hybrid faces */
        const PetscInt *cellFace = &cellFaces[-cf*faceSize];
        PetscInt       f;

        for (f = 0; f < -faceSize; ++f) sizes[cf] += (cellFace[f] >= 0 ? 1 : 0);
      }
      ierr = PetscSortInt(numCellFacesT, sizes);CHKERRQ(ierr);
      minv = sizes[0];
      maxv = sizes[PetscMax(numCellFacesT-1, 0)];
      if (minv != maxv) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Different number of vertices for non-hybrid face %D != %D", minv, maxv);
      faceSizeAll = minv;
      ierr = PetscMemzero(sizes, numCellFacesH*sizeof(PetscInt));CHKERRQ(ierr);
      for (cf = numCellFacesT; cf < numCellFacesH; ++cf) { /* These are the hybrid faces */
        const PetscInt *cellFace = &cellFaces[-cf*faceSize];
        PetscInt       f;

        for (f = 0; f < -faceSize; ++f) sizes[cf-numCellFacesT] += (cellFace[f] >= 0 ? 1 : 0);
      }
      ierr = PetscSortInt(numCellFacesH - numCellFacesT, sizes);CHKERRQ(ierr);
      minv = sizes[0];
      maxv = sizes[PetscMax(numCellFacesH - numCellFacesT-1, 0)];
      ierr = PetscFree(sizes);CHKERRQ(ierr);
      if (minv != maxv) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Different number of vertices for hybrid face %D != %D", minv, maxv);
      faceSizeAllH = minv;
    } else { /* the size of the faces in hybrid cells is the same */
      faceSizeAll = faceSizeAllH = faceSize;
    }
    ierr = DMPlexRestoreRawFacesHybrid_Internal(dm, cellDim, coneSizeH, cone, &numCellFacesH, &numCellFacesT, &faceSize, &cellFaces);CHKERRQ(ierr);
  } else if (pEnd[cellDepth] > pStart[cellDepth]) {
    ierr = DMPlexGetFaces_Internal(dm, cellDim, pStart[cellDepth], NULL, &faceSizeAll, NULL);CHKERRQ(ierr);
  }
  if (faceSizeAll > 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Do not support interpolation of meshes with faces of %D vertices", faceSizeAll);

  /* With hybrid grids, we first iterate on hybrid cells and start numbering the non-hybrid grids
     Then, faces for non-hybrid cells are numbered.
     This is to guarantee consistent orientations (all 0) of all the points in the cone of the hybrid cells */
  ierr = PetscHashIJKLCreate(&faceTable);CHKERRQ(ierr);
  for (outerloop = 0, face = pStart[faceDepth]; outerloop < 2; outerloop++) {
    PetscInt start, end;

    start = outerloop == 0 ? pMax : pStart[cellDepth];
    end = outerloop == 0 ? pEnd[cellDepth] : pMax;
    for (c = start; c < end; ++c) {
      const PetscInt *cellFaces;
      PetscInt        numCellFaces, faceSize, faceSizeInc, cf;

      if (c < pMax) {
        ierr = DMPlexGetFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
        if (faceSize != faceSizeAll) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent face for cell %D of size %D != %D", c, faceSize, faceSizeAll);
      } else { /* Hybrid cell */
        const PetscInt *cone;
        PetscInt        numCellFacesN, coneSize;

        ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
        if (coneSize != coneSizeH) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unexpected hybrid coneSize %D != %D", coneSize, coneSizeH);
        ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
        ierr = DMPlexGetRawFacesHybrid_Internal(dm, cellDim, coneSize, cone, &numCellFaces, &numCellFacesN, &faceSize, &cellFaces);CHKERRQ(ierr);
        if (numCellFaces != numCellFacesH) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unexpected numCellFaces %D != %D for hybrid cell %D", numCellFaces, numCellFacesH, c);
        faceSize = PetscMax(faceSize, -faceSize);
        if (faceSize > 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Do not support interpolation of meshes with faces of %D vertices", faceSize);
        numCellFaces = numCellFacesN; /* process only non-hybrid faces */
      }
      faceSizeInc = faceSize;
      for (cf = 0; cf < numCellFaces; ++cf) {
        const PetscInt   *cellFace = &cellFaces[cf*faceSizeInc];
        PetscInt          faceSizeH = faceSize;
        PetscHashIJKLKey  key;
        PetscHashIter     iter;
        PetscBool         missing;

        if (faceSizeInc == 2) {
          key.i = PetscMin(cellFace[0], cellFace[1]);
          key.j = PetscMax(cellFace[0], cellFace[1]);
          key.k = PETSC_MAX_INT;
          key.l = PETSC_MAX_INT;
        } else {
          key.i = cellFace[0];
          key.j = cellFace[1];
          key.k = cellFace[2];
          key.l = faceSize > 3 ? (cellFace[3] < 0 ? faceSizeH = 3, PETSC_MAX_INT : cellFace[3]) : PETSC_MAX_INT;
          ierr  = PetscSortInt(faceSize, (PetscInt *) &key);CHKERRQ(ierr);
        }
        /* this check is redundant for non-hybrid meshes */
        if (faceSizeH != faceSizeAll) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unexpected number of vertices for face %D of point %D -> %D != %D", cf, c, faceSizeH, faceSizeAll);
        ierr = PetscHashIJKLPut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
        if (missing) {ierr = PetscHashIJKLIterSet(faceTable, iter, face++);CHKERRQ(ierr);}
      }
      if (c < pMax) {
        ierr = DMPlexRestoreFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
      } else {
        ierr = DMPlexRestoreRawFacesHybrid_Internal(dm, cellDim, coneSizeH, NULL, NULL, NULL, NULL, &cellFaces);CHKERRQ(ierr);
      }
    }
  }
  pEnd[faceDepth] = face;

  /* Second pass for hybrid meshes: number hybrid faces */
  for (c = pMax; c < pEnd[cellDepth]; ++c) {
    const PetscInt *cellFaces, *cone;
    PetscInt        numCellFaces, numCellFacesN, faceSize, cf, coneSize;

    ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetRawFacesHybrid_Internal(dm, cellDim, coneSize, cone, &numCellFaces, &numCellFacesN, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (numCellFaces != numCellFacesH) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unexpected hybrid numCellFaces %D != %D", numCellFaces, numCellFacesH);
    faceSize = PetscMax(faceSize, -faceSize);
    for (cf = numCellFacesN; cf < numCellFaces; ++cf) { /* These are the hybrid faces */
      const PetscInt   *cellFace = &cellFaces[cf*faceSize];
      PetscHashIJKLKey  key;
      PetscHashIter     iter;
      PetscBool         missing;
      PetscInt          faceSizeH = faceSize;

      if (faceSize == 2) {
        key.i = PetscMin(cellFace[0], cellFace[1]);
        key.j = PetscMax(cellFace[0], cellFace[1]);
        key.k = PETSC_MAX_INT;
        key.l = PETSC_MAX_INT;
      } else {
        key.i = cellFace[0];
        key.j = cellFace[1];
        key.k = cellFace[2];
        key.l = faceSize > 3 ? (cellFace[3] < 0 ? faceSizeH = 3, PETSC_MAX_INT : cellFace[3]) : PETSC_MAX_INT;
        ierr  = PetscSortInt(faceSize, (PetscInt *) &key);CHKERRQ(ierr);
      }
      if (faceSizeH != faceSizeAllH) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unexpected number of vertices for hybrid face %D of point %D -> %D != %D", cf, c, faceSizeH, faceSizeAllH);
      ierr = PetscHashIJKLPut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
      if (missing) {ierr = PetscHashIJKLIterSet(faceTable, iter, face++);CHKERRQ(ierr);}
    }
    ierr = DMPlexRestoreRawFacesHybrid_Internal(dm, cellDim, coneSize, cone, &numCellFaces, &numCellFacesN, &faceSize, &cellFaces);CHKERRQ(ierr);
  }
  faceH = face - pEnd[faceDepth];
  if (faceH) {
    if (fMax == PETSC_DETERMINE) fMax = pEnd[faceDepth];
    else if (eMax == PETSC_DETERMINE) eMax = pEnd[faceDepth];
    else SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of unassigned hybrid facets %D for cellDim %D and dimension %D", faceH, cellDim, dim);
  }
  pEnd[faceDepth] = face;
  ierr = PetscHashIJKLDestroy(&faceTable);CHKERRQ(ierr);
  /* Count new points */
  for (d = 0; d <= depth; ++d) {
    numPoints += pEnd[d]-pStart[d];
  }
  ierr = DMPlexSetChart(idm, 0, numPoints);CHKERRQ(ierr);
  /* Set cone sizes */
  for (d = 0; d <= depth; ++d) {
    PetscInt coneSize, p;

    if (d == faceDepth) {
      /* I see no way to do this if we admit faces of different shapes */
      for (p = pStart[d]; p < pEnd[d]-faceH; ++p) {
        ierr = DMPlexSetConeSize(idm, p, faceSizeAll);CHKERRQ(ierr);
      }
      for (p = pEnd[d]-faceH; p < pEnd[d]; ++p) {
        ierr = DMPlexSetConeSize(idm, p, faceSizeAllH);CHKERRQ(ierr);
      }
    } else if (d == cellDepth) {
      for (p = pStart[d]; p < pEnd[d]; ++p) {
        /* Number of cell faces may be different from number of cell vertices*/
        if (p < pMax) {
          ierr = DMPlexGetFaces_Internal(dm, cellDim, p, &coneSize, NULL, NULL);CHKERRQ(ierr);
        } else {
          ierr = DMPlexGetFacesHybrid_Internal(dm, cellDim, p, &coneSize, NULL, NULL, NULL);CHKERRQ(ierr);
        }
        ierr = DMPlexSetConeSize(idm, p, coneSize);CHKERRQ(ierr);
      }
    } else {
      for (p = pStart[d]; p < pEnd[d]; ++p) {
        ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexSetConeSize(idm, p, coneSize);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMSetUp(idm);CHKERRQ(ierr);
  /* Get face cones from subsets of cell vertices */
  if (faceSizeAll > 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Do not support interpolation of meshes with faces of %D vertices", faceSizeAll);
  ierr = PetscHashIJKLCreate(&faceTable);CHKERRQ(ierr);
  for (d = depth; d > cellDepth; --d) {
    const PetscInt *cone;
    PetscInt        p;

    for (p = pStart[d]; p < pEnd[d]; ++p) {
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexSetCone(idm, p, cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(idm, p, cone);CHKERRQ(ierr);
    }
  }
  for (outerloop = 0, face = pStart[faceDepth]; outerloop < 2; outerloop++) {
    PetscInt start, end;

    start = outerloop == 0 ? pMax : pStart[cellDepth];
    end = outerloop == 0 ? pEnd[cellDepth] : pMax;
    for (c = start; c < end; ++c) {
      const PetscInt *cellFaces;
      PetscInt        numCellFaces, faceSize, faceSizeInc, cf;

      if (c < pMax) {
        ierr = DMPlexGetFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
        if (faceSize != faceSizeAll) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent face for cell %D of size %D != %D", c, faceSize, faceSizeAll);
      } else {
        const PetscInt *cone;
        PetscInt        numCellFacesN, coneSize;

        ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
        if (coneSize != coneSizeH) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unexpected hybrid coneSize %D != %D", coneSize, coneSizeH);
        ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
        ierr = DMPlexGetRawFacesHybrid_Internal(dm, cellDim, coneSize, cone, &numCellFaces, &numCellFacesN, &faceSize, &cellFaces);CHKERRQ(ierr);
        if (numCellFaces != numCellFacesH) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unexpected numCellFaces %D != %D for hybrid cell %D", numCellFaces, numCellFacesH, c);
        faceSize = PetscMax(faceSize, -faceSize);
        if (faceSize > 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Do not support interpolation of meshes with faces of %D vertices", faceSize);
        numCellFaces = numCellFacesN; /* process only non-hybrid faces */
      }
      faceSizeInc = faceSize;
      for (cf = 0; cf < numCellFaces; ++cf) {
        const PetscInt  *cellFace = &cellFaces[cf*faceSizeInc];
        PetscHashIJKLKey key;
        PetscHashIter    iter;
        PetscBool        missing;

        if (faceSizeInc == 2) {
          key.i = PetscMin(cellFace[0], cellFace[1]);
          key.j = PetscMax(cellFace[0], cellFace[1]);
          key.k = PETSC_MAX_INT;
          key.l = PETSC_MAX_INT;
        } else {
          key.i = cellFace[0];
          key.j = cellFace[1];
          key.k = cellFace[2];
          key.l = faceSizeInc > 3 ? (cellFace[3] < 0 ? faceSize = 3, PETSC_MAX_INT : cellFace[3]) : PETSC_MAX_INT;
          ierr  = PetscSortInt(faceSizeInc, (PetscInt *) &key);CHKERRQ(ierr);
        }
        ierr = PetscHashIJKLPut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
        if (missing) {
          ierr = DMPlexSetCone(idm, face, cellFace);CHKERRQ(ierr);
          ierr = PetscHashIJKLIterSet(faceTable, iter, face);CHKERRQ(ierr);
          ierr = DMPlexInsertCone(idm, c, cf, face++);CHKERRQ(ierr);
        } else {
          const PetscInt *cone;
          PetscInt        coneSize, ornt, i, j, f;

          ierr = PetscHashIJKLIterGet(faceTable, iter, &f);CHKERRQ(ierr);
          ierr = DMPlexInsertCone(idm, c, cf, f);CHKERRQ(ierr);
          /* Orient face: Do not allow reverse orientation at the first vertex */
          ierr = DMPlexGetConeSize(idm, f, &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(idm, f, &cone);CHKERRQ(ierr);
          if (coneSize != faceSize) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of face vertices %D for face %D should be %D", coneSize, f, faceSize);
          /* - First find the initial vertex */
          for (i = 0; i < faceSize; ++i) if (cellFace[0] == cone[i]) break;
          /* - Try forward comparison */
          for (j = 0; j < faceSize; ++j) if (cellFace[j] != cone[(i+j)%faceSize]) break;
          if (j == faceSize) {
            if ((faceSize == 2) && (i == 1)) ornt = -2;
            else                             ornt = i;
          } else {
            /* - Try backward comparison */
            for (j = 0; j < faceSize; ++j) if (cellFace[j] != cone[(i+faceSize-j)%faceSize]) break;
            if (j == faceSize) {
              if (i == 0) ornt = -faceSize;
              else        ornt = -i;
            } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not determine face orientation");
          }
          ierr = DMPlexInsertConeOrientation(idm, c, cf, ornt);CHKERRQ(ierr);
        }
      }
      if (c < pMax) {
        ierr = DMPlexRestoreFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
      } else {
        ierr = DMPlexRestoreRawFacesHybrid_Internal(dm, cellDim, coneSizeH, NULL, NULL, NULL, NULL, &cellFaces);CHKERRQ(ierr);
      }
    }
  }
  /* Second pass for hybrid meshes: orient hybrid faces */
  for (c = pMax; c < pEnd[cellDepth]; ++c) {
    const PetscInt *cellFaces, *cone;
    PetscInt        numCellFaces, numCellFacesN, faceSize, cf, coneSize;

    ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetRawFacesHybrid_Internal(dm, cellDim, coneSize, cone, &numCellFaces, &numCellFacesN, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (numCellFaces != numCellFacesH) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unexpected hybrid numCellFaces %D != %D", numCellFaces, numCellFacesH);
    faceSize = PetscMax(faceSize, -faceSize);
    for (cf = numCellFacesN; cf < numCellFaces; ++cf) { /* These are the hybrid faces */
      const PetscInt   *cellFace = &cellFaces[cf*faceSize];
      PetscHashIJKLKey key;
      PetscHashIter    iter;
      PetscBool        missing;
      PetscInt         faceSizeH = faceSize;

      if (faceSize == 2) {
        key.i = PetscMin(cellFace[0], cellFace[1]);
        key.j = PetscMax(cellFace[0], cellFace[1]);
        key.k = PETSC_MAX_INT;
        key.l = PETSC_MAX_INT;
      } else {
        key.i = cellFace[0];
        key.j = cellFace[1];
        key.k = cellFace[2];
        key.l = faceSize > 3 ? (cellFace[3] < 0 ? faceSizeH = 3, PETSC_MAX_INT : cellFace[3]) : PETSC_MAX_INT;
        ierr  = PetscSortInt(faceSize, (PetscInt *) &key);CHKERRQ(ierr);
      }
      if (faceSizeH != faceSizeAllH) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unexpected number of vertices for hybrid face %D of point %D -> %D != %D", cf, c, faceSizeH, faceSizeAllH);
      ierr = PetscHashIJKLPut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
      if (missing) {
        ierr = DMPlexSetCone(idm, face, cellFace);CHKERRQ(ierr);
        ierr = PetscHashIJKLIterSet(faceTable, iter, face);CHKERRQ(ierr);
        ierr = DMPlexInsertCone(idm, c, cf, face++);CHKERRQ(ierr);
      } else {
        const PetscInt *cone;
        PetscInt        coneSize, ornt, i, j, f;

        ierr = PetscHashIJKLIterGet(faceTable, iter, &f);CHKERRQ(ierr);
        ierr = DMPlexInsertCone(idm, c, cf, f);CHKERRQ(ierr);
        /* Orient face: Do not allow reverse orientation at the first vertex */
        ierr = DMPlexGetConeSize(idm, f, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(idm, f, &cone);CHKERRQ(ierr);
        if (coneSize != faceSizeH) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of face vertices %D for face %D should be %D", coneSize, f, faceSizeH);
        /* - First find the initial vertex */
        for (i = 0; i < faceSizeH; ++i) if (cellFace[0] == cone[i]) break;
        /* - Try forward comparison */
        for (j = 0; j < faceSizeH; ++j) if (cellFace[j] != cone[(i+j)%faceSizeH]) break;
        if (j == faceSizeH) {
          if ((faceSizeH == 2) && (i == 1)) ornt = -2;
          else                             ornt = i;
        } else {
          /* - Try backward comparison */
          for (j = 0; j < faceSizeH; ++j) if (cellFace[j] != cone[(i+faceSizeH-j)%faceSizeH]) break;
          if (j == faceSizeH) {
            if (i == 0) ornt = -faceSizeH;
            else        ornt = -i;
          } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not determine face orientation");
        }
        ierr = DMPlexInsertConeOrientation(idm, c, cf, ornt);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreRawFacesHybrid_Internal(dm, cellDim, coneSize, cone, &numCellFaces, &numCellFacesN, &faceSize, &cellFaces);CHKERRQ(ierr);
  }
  if (face != pEnd[faceDepth]) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of faces %D should be %D", face-pStart[faceDepth], pEnd[faceDepth]-pStart[faceDepth]);
  ierr = PetscFree2(pStart,pEnd);CHKERRQ(ierr);
  ierr = PetscHashIJKLDestroy(&faceTable);CHKERRQ(ierr);
  ierr = PetscFree2(pStart,pEnd);CHKERRQ(ierr);
  ierr = DMPlexSetHybridBounds(idm, cMax, fMax, eMax, vMax);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(idm);CHKERRQ(ierr);
  ierr = DMPlexStratify(idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This interpolates the PointSF in parallel following local interpolation */
static PetscErrorCode DMPlexInterpolatePointSF(DM dm, PetscSF pointSF, PetscInt depth)
{
  PetscMPIInt        size, rank;
  PetscInt           p, c, d, dof, offset;
  PetscInt           numLeaves, numRoots, candidatesSize, candidatesRemoteSize;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscSFNode       *candidates, *candidatesRemote, *claims;
  PetscSection       candidateSection, candidateSectionRemote, claimSection;
  PetscHMapI         leafhash;
  PetscHMapIJ        roothash;
  PetscHashIJKey     key;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(pointSF, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (size < 2 || numRoots < 0) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(DMPLEX_InterpolateSF,dm,0,0,0);CHKERRQ(ierr);
  /* Build hashes of points in the SF for efficient lookup */
  ierr = PetscHMapICreate(&leafhash);CHKERRQ(ierr);
  ierr = PetscHMapIJCreate(&roothash);CHKERRQ(ierr);
  for (p = 0; p < numLeaves; ++p) {
    ierr = PetscHMapISet(leafhash, localPoints[p], p);CHKERRQ(ierr);
    key.i = remotePoints[p].index;
    key.j = remotePoints[p].rank;
    ierr = PetscHMapIJSet(roothash, key, p);CHKERRQ(ierr);
  }
  /* Build a section / SFNode array of candidate points in the single-level adjacency of leaves,
     where each candidate is defined by the root entry for the other vertex that defines the edge. */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &candidateSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(candidateSection, 0, numRoots);CHKERRQ(ierr);
  {
    PetscInt leaf, root, idx, a, *adj = NULL;
    for (p = 0; p < numLeaves; ++p) {
      PetscInt adjSize = PETSC_DETERMINE;
      ierr = DMPlexGetAdjacency_Internal(dm, localPoints[p], PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &adjSize, &adj);CHKERRQ(ierr);
      for (a = 0; a < adjSize; ++a) {
        ierr = PetscHMapIGet(leafhash, adj[a], &leaf);CHKERRQ(ierr);
        if (leaf >= 0) {ierr = PetscSectionAddDof(candidateSection, localPoints[p], 1);CHKERRQ(ierr);}
      }
    }
    ierr = PetscSectionSetUp(candidateSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(candidateSection, &candidatesSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(candidatesSize, &candidates);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      PetscInt adjSize = PETSC_DETERMINE;
      ierr = PetscSectionGetOffset(candidateSection, localPoints[p], &offset);CHKERRQ(ierr);
      ierr = DMPlexGetAdjacency_Internal(dm, localPoints[p], PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &adjSize, &adj);CHKERRQ(ierr);
      for (idx = 0, a = 0; a < adjSize; ++a) {
        ierr = PetscHMapIGet(leafhash, adj[a], &root);CHKERRQ(ierr);
        if (root >= 0) candidates[offset+idx++] = remotePoints[root];
      }
    }
    ierr = PetscFree(adj);CHKERRQ(ierr);
  }
  /* Gather candidate section / array pair into the root partition via inverse(multi(pointSF)). */
  {
    PetscSF   sfMulti, sfInverse, sfCandidates;
    PetscInt *remoteOffsets;
    ierr = PetscSFGetMultiSF(pointSF, &sfMulti);CHKERRQ(ierr);
    ierr = PetscSFCreateInverseSF(sfMulti, &sfInverse);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &candidateSectionRemote);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfInverse, candidateSection, &remoteOffsets, candidateSectionRemote);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(sfInverse, candidateSection, remoteOffsets, candidateSectionRemote, &sfCandidates);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(candidateSectionRemote, &candidatesRemoteSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(candidatesRemoteSize, &candidatesRemote);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sfCandidates, MPIU_2INT, candidates, candidatesRemote);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfCandidates, MPIU_2INT, candidates, candidatesRemote);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfInverse);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfCandidates);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  }
  /* Walk local roots and check for each remote candidate whether we know all required points,
     either from owning it or having a root entry in the point SF. If we do we place a claim
     by replacing the vertex number with our edge ID. */
  {
    PetscInt        idx, root, joinSize, vertices[2];
    const PetscInt *rootdegree, *join = NULL;
    ierr = PetscSFComputeDegreeBegin(pointSF, &rootdegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(pointSF, &rootdegree);CHKERRQ(ierr);
    /* Loop remote edge connections and put in a claim if both vertices are known */
    for (idx = 0, p = 0; p < numRoots; ++p) {
      for (d = 0; d < rootdegree[p]; ++d) {
        ierr = PetscSectionGetDof(candidateSectionRemote, idx, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(candidateSectionRemote, idx, &offset);CHKERRQ(ierr);
        for (c = 0; c < dof; ++c) {
          /* We own both vertices, so we claim the edge by replacing vertex with edge */
          if (candidatesRemote[offset+c].rank == rank) {
            vertices[0] = p; vertices[1] = candidatesRemote[offset+c].index;
            ierr = DMPlexGetJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
            if (joinSize == 1) candidatesRemote[offset+c].index = join[0];
            ierr = DMPlexRestoreJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
            continue;
          }
          /* If we own one vertex and share a root with the other, we claim it */
          key.i = candidatesRemote[offset+c].index;
          key.j = candidatesRemote[offset+c].rank;
          ierr = PetscHMapIJGet(roothash, key, &root);CHKERRQ(ierr);
          if (root >= 0) {
            vertices[0] = p; vertices[1] = localPoints[root];
            ierr = DMPlexGetJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
            if (joinSize == 1) {
              candidatesRemote[offset+c].index = join[0];
              candidatesRemote[offset+c].rank = rank;
            }
            ierr = DMPlexRestoreJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
          }
        }
        idx++;
      }
    }
  }
  /* Push claims back to receiver via the MultiSF and derive new pointSF mapping on receiver */
  {
    PetscSF         sfMulti, sfClaims, sfPointNew;
    PetscHMapI      claimshash;
    PetscInt        size, pStart, pEnd, root, joinSize, numLocalNew;
    PetscInt       *remoteOffsets, *localPointsNew, vertices[2];
    const PetscInt *join = NULL;
    PetscSFNode    *remotePointsNew;
    ierr = PetscSFGetMultiSF(pointSF, &sfMulti);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &claimSection);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfMulti, candidateSectionRemote, &remoteOffsets, claimSection);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(sfMulti, candidateSectionRemote, remoteOffsets, claimSection, &sfClaims);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(claimSection, &size);CHKERRQ(ierr);
    ierr = PetscMalloc1(size, &claims);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sfClaims, MPIU_2INT, candidatesRemote, claims);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfClaims, MPIU_2INT, candidatesRemote, claims);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfClaims);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
    /* Walk the original section of local supports and add an SF entry for each updated item */
    ierr = PetscHMapICreate(&claimshash);CHKERRQ(ierr);
    for (p = 0; p < numRoots; ++p) {
      ierr = PetscSectionGetDof(candidateSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(candidateSection, p, &offset);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        if (candidates[offset+d].index != claims[offset+d].index) {
          key.i = candidates[offset+d].index;
          key.j = candidates[offset+d].rank;
          ierr = PetscHMapIJGet(roothash, key, &root);CHKERRQ(ierr);
          if (root >= 0) {
            vertices[0] = p; vertices[1] = localPoints[root];
            ierr = DMPlexGetJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
            if (joinSize == 1) {ierr = PetscHMapISet(claimshash, join[0], offset+d);CHKERRQ(ierr);}
            ierr = DMPlexRestoreJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
          }
        }
      }
    }
    /* Create new pointSF from hashed claims */
    ierr = PetscHMapIGetSize(claimshash, &numLocalNew);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeaves + numLocalNew, &localPointsNew);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeaves + numLocalNew, &remotePointsNew);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      localPointsNew[p] = localPoints[p];
      remotePointsNew[p].index = remotePoints[p].index;
      remotePointsNew[p].rank  = remotePoints[p].rank;
    }
    p = numLeaves;
    ierr = PetscHMapIGetKeys(claimshash, &p, localPointsNew);CHKERRQ(ierr);
    ierr = PetscSortInt(numLocalNew, &localPointsNew[numLeaves]);CHKERRQ(ierr);
    for (p = numLeaves; p < numLeaves + numLocalNew; ++p) {
      ierr = PetscHMapIGet(claimshash, localPointsNew[p], &offset);CHKERRQ(ierr);
      remotePointsNew[p] = claims[offset];
    }
    ierr = PetscSFCreate(PetscObjectComm((PetscObject) dm), &sfPointNew);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfPointNew, pEnd-pStart, numLeaves+numLocalNew, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = DMSetPointSF(dm, sfPointNew);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfPointNew);CHKERRQ(ierr);
    ierr = PetscHMapIDestroy(&claimshash);CHKERRQ(ierr);
  }
  ierr = PetscHMapIDestroy(&leafhash);CHKERRQ(ierr);
  ierr = PetscHMapIJDestroy(&roothash);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&candidateSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&candidateSectionRemote);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&claimSection);CHKERRQ(ierr);
  ierr = PetscFree(candidates);CHKERRQ(ierr);
  ierr = PetscFree(candidatesRemote);CHKERRQ(ierr);
  ierr = PetscFree(claims);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_InterpolateSF,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define PetscSectionExpandPoints_Loop(TYPE) \
{ \
  PetscInt i, n, o0, o1; \
  TYPE *a0 = (TYPE*)origArray, *a1; \
  ierr = PetscMalloc1(size, &a1);CHKERRQ(ierr); \
  for (i=0; i<npoints; i++) { \
    ierr = PetscSectionGetOffset(origSection, points_[i], &o0);CHKERRQ(ierr); \
    ierr = PetscSectionGetOffset(s, i, &o1);CHKERRQ(ierr); \
    ierr = PetscSectionGetDof(s, i, &n);CHKERRQ(ierr); \
    ierr = PetscMemcpy(&a1[o1], &a0[o0], n*unitsize);CHKERRQ(ierr); \
  } \
  *newArray = (void*)a1; \
}

/* TODO add to PetscSection API */
PetscErrorCode PetscSectionExpandPoints(PetscSection origSection, MPI_Datatype dataType, const void *origArray, IS points, PetscInt *newSize, PetscSection *newSection, void *newArray[])
{
  PetscSection        s;
  const PetscInt      *points_;
  PetscInt            i, n, npoints, pStart, pEnd, size;
  PetscMPIInt         unitsize;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MPI_Type_size(dataType, &unitsize);CHKERRQ(ierr);
  ierr = ISGetLocalSize(points, &npoints);CHKERRQ(ierr);
  ierr = ISGetIndices(points, &points_);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(origSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &s);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(s, 0, npoints);CHKERRQ(ierr);
  for (i=0; i<npoints; i++) {
    if (PetscUnlikely(points_[i] < pStart || points_[i] >= pEnd)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "point %d (index %d) in input IS out of input section's chart", points_[i], i);
    ierr = PetscSectionGetDof(origSection, points_[i], &n);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(s, i, n);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(s, &size);CHKERRQ(ierr);
  if (newArray) {
    switch (dataType) {
      case MPIU_INT:      PetscSectionExpandPoints_Loop(PetscInt); break;
      case MPIU_SCALAR:   PetscSectionExpandPoints_Loop(PetscScalar); break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "not implemented for datatype %d", dataType);
    }
  }
  if (newSection) {
    *newSection = s;
  } else {
    ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  }
  if (newSize) *newSize = size;
  PetscFunctionReturn(0);
}

/* TODO add to DMPlex API */
PetscErrorCode DMPlexGetCoordinatesTuple(DM dm, IS points, PetscSection *pCoordSection, Vec *pCoord)
{
  PetscSection        cs;
  Vec                 coords;
  const PetscScalar   *arr;
  PetscScalar         *newarr=NULL;
  PetscInt            n;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(points, IS_CLASSID, 2);
  if (pCoordSection) PetscValidPointer(pCoordSection, 3);
  if (pCoord) PetscValidPointer(pCoord, 4);
  ierr = DMGetCoordinateSection(dm, &cs);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &arr);CHKERRQ(ierr);
  ierr = PetscSectionExpandPoints(cs, MPIU_SCALAR, arr, points, &n, pCoordSection, pCoord ? ((void**)&newarr) : NULL);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coords, &arr);CHKERRQ(ierr);
  if (pCoord) {
    /* set array in two steps to mimic PETSC_OWN_POINTER */
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)points), 1, n, NULL, pCoord);CHKERRQ(ierr);
    ierr = VecReplaceArray(*pCoord, newarr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* TODO add to DMPlex API */
PetscErrorCode DMPlexGetConesTuple(DM dm, IS points, PetscSection *pConesSection, IS *pCones)
{
  PetscSection        cs;
  PetscInt            *cones;
  PetscInt            *newarr=NULL;
  PetscInt            n;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(dm, &cs);CHKERRQ(ierr);
  ierr = PetscSectionExpandPoints(cs, MPIU_INT, cones, points, &n, pConesSection, pCones ? ((void**)&newarr) : NULL);CHKERRQ(ierr);
  if (pCones) {
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)points), n, newarr, PETSC_OWN_POINTER, pCones);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGetConeRecursive_Private(DM dm, PetscInt *n_inout, const PetscInt points[], PetscInt *offset_inout, PetscInt buf[])
{
  PetscInt p, n, cn, i;
  const PetscInt *cone;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n = *n_inout;
  *n_inout = 0;
  for (i=0; i<n; i++) {
    p = points[i];
    ierr = DMPlexGetConeSize(dm, p, &cn);CHKERRQ(ierr);
    if (!cn) {
      cn = 1;
      if (buf) {
        buf[*offset_inout] = p;
        ++(*offset_inout);
      }
    } else {
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeRecursive_Private(dm, &cn, cone, offset_inout, buf);CHKERRQ(ierr);
    }
    *n_inout += cn;
  }
  PetscFunctionReturn(0);
}

/* TODO add to DMPlex API */
PetscErrorCode DMPlexGetConesRecursive(DM dm, IS points, IS *pCones)
{
  const PetscInt      *arr=NULL;
  /* TODO this should be const */
  PetscInt            *cpoints=NULL;
  PetscInt            n, cn;
  PetscInt            zero;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(points, &n);CHKERRQ(ierr);
  ierr = ISGetIndices(points, &arr);CHKERRQ(ierr);
  zero = 0;
  /* first figure out the total size */
  cn = n;
  ierr = DMPlexGetConeRecursive_Private(dm, &cn, arr, &zero, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(cn, &cpoints);CHKERRQ(ierr);
  /* now get recursive cones for real */
  ierr = DMPlexGetConeRecursive_Private(dm, &n, arr, &zero, cpoints);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, n, cpoints, PETSC_OWN_POINTER, pCones);CHKERRQ(ierr);
  ierr = ISRestoreIndices(points, &arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Compare cones of the master and slave face (with the same cone points modulo order), and return relative orientation of the slave. */
PETSC_STATIC_INLINE PetscErrorCode DMPlexFixFaceOrientations_Orient_Private(PetscInt coneSize, const PetscInt masterCone[], const PetscInt slaveCone[], PetscInt *start, PetscBool *reverse)
{
  PetscInt        i;

  PetscFunctionBegin;
  for (i=0; i<coneSize; i++) {
    if (slaveCone[i] == masterCone[0]) {
      *start = i;
      break;
    }
  }
  if (PetscUnlikely(i==coneSize)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "starting point of master cone not found in slave cone");
  *reverse = PETSC_FALSE;
  for (i=0; i<coneSize; i++) {if (slaveCone[((*start)+i)%coneSize] != masterCone[i]) break;}
  if (i == coneSize) PetscFunctionReturn(0);
  *reverse = PETSC_TRUE;
  for (i=0; i<coneSize; i++) {if (slaveCone[(coneSize+(*start)-i)%coneSize] != masterCone[i]) break;}
  if (i < coneSize) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "master and slave cone have non-conforming order of points");
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMPlexFixFaceOrientations_Translate_Private(PetscInt ornt, PetscInt *start, PetscBool *reverse)
{
  PetscFunctionBegin;
  *reverse = (ornt < 0) ? PETSC_TRUE : PETSC_FALSE;
  *start = *reverse ? -(ornt+1) : ornt;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMPlexFixFaceOrientations_Combine_Private(PetscInt coneSize, PetscInt start0, PetscBool reverse0, PetscInt start1, PetscBool reverse1, PetscInt *start, PetscBool *reverse)
{
  PetscFunctionBegin;
  *reverse = (reverse0 == reverse1) ? PETSC_FALSE : PETSC_TRUE;
  *start = ((start0 + start1) % coneSize);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMPlexFixFaceOrientations_TranslateBack_Private(PetscInt coneSize, PetscInt start, PetscBool reverse, PetscInt *ornt)
{
  PetscFunctionBegin;
  if (coneSize < 3) {
    /* edges just get flipped */
    *ornt = start ? -2 : 0;
  } else {
    *ornt = reverse ? -(start+1) : start;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexFixFaceOrientations_Private(DM dm, IS points, PetscSection coneSection, IS correctCones, IS wrongCones)
{
  PetscInt i, j, k, n, o, p, q, pStart, pEnd, coneSize, supportSize, supportConeSize;
  PetscInt start0, start1, start;
  PetscBool reverse0, reverse1, reverse;
  PetscInt newornt;
  const PetscInt *correctCones_, *wrongCones_, *points_, *support, *supportCone, *ornts;
  PetscInt *newornts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetIndices(points, &points_);CHKERRQ(ierr);
  ierr = ISGetIndices(correctCones, &correctCones_);CHKERRQ(ierr);
  ierr = ISGetIndices(wrongCones, &wrongCones_);CHKERRQ(ierr);

  {
    PetscMPIInt myrank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"[%d] BEGIN DMPlexFixFaceOrientations_Private\n", myrank);
  }
  ierr = PetscSectionGetChart(coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  for (p=pStart; p<pEnd; p++) {
    ierr = PetscSectionGetDof(coneSection, p, &n);CHKERRQ(ierr);
    if (!n) continue; /* do nothing for points with no cone */
    ierr = PetscSectionGetOffset(coneSection, p, &o);CHKERRQ(ierr);
    ierr = DMPlexFixFaceOrientations_Orient_Private(n, &correctCones_[o], &wrongCones_[o], &start1, &reverse1);CHKERRQ(ierr);
    {
      PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"##### p=%d o=%d point[p]=%d &wrongCones_[o] &correctCones_[o]\n", p, o, points_[p]);
      PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_SELF);
      PetscIntView(n, &wrongCones_[o], PETSC_VIEWER_STDOUT_SELF);
      PetscIntView(n, &correctCones_[o], PETSC_VIEWER_STDOUT_SELF);
    }
    if (start1 || reverse1) {
      q = points_[p];
      ierr = DMPlexGetConeSize(dm, q, &coneSize);CHKERRQ(ierr);
      /* permute q's cone orientations */
      ierr = DMPlexGetConeOrientation(dm, q, &ornts);CHKERRQ(ierr);
      ierr = PetscMalloc1(coneSize, &newornts);CHKERRQ(ierr);
      if (reverse1) {for (i=0; i<coneSize; i++) newornts[(coneSize+start1-i)%coneSize] = ornts[i];}
      else          {for (i=0; i<coneSize; i++) newornts[(start1+i)%coneSize] = ornts[i];}
      {
        PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"ornts newornts\n");
        PetscIntView(coneSize, ornts, PETSC_VIEWER_STDOUT_SELF);
        PetscIntView(coneSize, newornts, PETSC_VIEWER_STDOUT_SELF);
      }
      ierr = DMPlexSetConeOrientation(dm, q, newornts);CHKERRQ(ierr);
      ierr = PetscFree(newornts);CHKERRQ(ierr);
      /* fix oriention of q within cones of q's support points */
      ierr = DMPlexGetSupport(dm, q, &support);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, q, &supportSize);CHKERRQ(ierr);
      for (j=0; j<supportSize; j++) {
        ierr = DMPlexGetCone(dm, support[j], &supportCone);CHKERRQ(ierr);
        ierr = DMPlexGetConeSize(dm, support[j], &supportConeSize);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, support[j], &ornts);CHKERRQ(ierr);
        for (k=0; k<supportConeSize; k++) {
          if (supportCone[k] == q) {
            ierr = DMPlexFixFaceOrientations_Translate_Private(ornts[k], &start0, &reverse0);CHKERRQ(ierr);
            ierr = DMPlexFixFaceOrientations_Combine_Private(coneSize, start0, reverse0, start1, reverse1, &start, &reverse);CHKERRQ(ierr);
            ierr = DMPlexFixFaceOrientations_TranslateBack_Private(coneSize, start, reverse, &newornt);CHKERRQ(ierr);
            {
              PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"interface point %d has original orientation %d within cone of %d (cone local index %d of %d)\n", q, ornts[k], support[j], k, supportConeSize);
              PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"  start0 %d start1 %d start %d\n", start0, start1, start);
              PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"  reverse0 %d reverse1 %d reverse %d\n", reverse0, reverse1, reverse);
              PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"  new orientation %d\n", newornt);
            }
            ierr = DMPlexInsertConeOrientation(dm, support[j], k, newornt);CHKERRQ(ierr);
          }
        }
      }
      /* rewrite cone */
      ierr = DMPlexSetCone(dm, q, &correctCones_[o]);CHKERRQ(ierr);
    }
    {
      PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_SELF);
    }
  }

  {
    PetscMPIInt myrank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"[%d] END DMPlexFixFaceOrientations_Private\n", myrank);
  }

  ierr = ISRestoreIndices(points, &points_);CHKERRQ(ierr);
  ierr = ISRestoreIndices(correctCones, &correctCones_);CHKERRQ(ierr);
  ierr = ISRestoreIndices(wrongCones, &wrongCones_);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TODO PetscArrayExchangeBegin/End */
/* TODO blocksize */
static PetscErrorCode ExchangeArrayByRank_Private(PetscObject obj, MPI_Datatype dt, PetscInt nsranks, const PetscMPIInt sranks[], PetscInt ssize[], const void *sarr[], PetscInt nrranks, const PetscMPIInt rranks[], PetscInt *rsize_out[], void **rarr_out[])
{
  PetscInt r;
  PetscInt *rsize;
  void **rarr;
  MPI_Request *sreq, *rreq;
  PetscMPIInt tag, unitsize;
  MPI_Comm comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Type_size(dt, &unitsize);CHKERRQ(ierr);
  ierr = PetscObjectGetComm(obj, &comm);CHKERRQ(ierr);
  ierr = PetscMalloc2(nrranks, &rsize, nrranks, &rarr);CHKERRQ(ierr);
  ierr = PetscMalloc2(nrranks, &rreq, nsranks, &sreq);CHKERRQ(ierr);
  /* exchange array size */
  ierr = PetscObjectGetNewTag(obj,&tag);CHKERRQ(ierr);
  for (r=0; r<nrranks; r++) {
    ierr = MPI_Irecv(&rsize[r], 1, MPIU_INT, rranks[r], tag, comm, &rreq[r]);CHKERRQ(ierr);
  }
  for (r=0; r<nsranks; r++) {
    ierr = MPI_Isend(&ssize[r], 1, MPIU_INT, sranks[r], tag, comm, &sreq[r]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(nrranks, rreq, MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  /* exchange array */
  ierr = PetscObjectGetNewTag(obj,&tag);CHKERRQ(ierr);
  for (r=0; r<nrranks; r++) {
    ierr = PetscMalloc(rsize[r]*unitsize, &rarr[r]);CHKERRQ(ierr);
    ierr = MPI_Irecv(rarr[r], rsize[r], dt, rranks[r], tag, comm, &rreq[r]);CHKERRQ(ierr);
  }
  for (r=0; r<nsranks; r++) {
    ierr = MPI_Isend(sarr[r], ssize[r], dt, sranks[r], tag, comm, &sreq[r]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(nrranks, rreq, MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(nsranks, sreq, MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree2(rreq, sreq);CHKERRQ(ierr);
  *rsize_out = rsize;
  *rarr_out = rarr;
  PetscFunctionReturn(0);
}

/* TODO ISExchangeBegin/End */
static PetscErrorCode ExchangeISByRank_Private(PetscObject obj, PetscInt nsranks, const PetscMPIInt sranks[], IS sis[], PetscInt nrranks, const PetscMPIInt rranks[], IS *ris[])
{
  PetscInt r;
  PetscInt *ssize, *rsize;
  PetscInt **rarr;
  const PetscInt **sarr;
  IS *ris_;
  MPI_Request *sreq, *rreq;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc4(nsranks, &ssize, nsranks, &sarr, nrranks, &rreq, nsranks, &sreq);CHKERRQ(ierr);
  for (r=0; r<nsranks; r++) {
    ierr = ISGetLocalSize(sis[r], &ssize[r]);CHKERRQ(ierr);
    ierr = ISGetIndices(sis[r], &sarr[r]);CHKERRQ(ierr);
  }
  ierr = ExchangeArrayByRank_Private(obj, MPIU_INT, nsranks, sranks, ssize, (const void**)sarr, nrranks, rranks, &rsize, (void***)&rarr);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrranks, &ris_);CHKERRQ(ierr);
  for (r=0; r<nrranks; r++) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF, rsize[r], rarr[r], PETSC_OWN_POINTER, &ris_[r]);CHKERRQ(ierr);
  }
  for (r=0; r<nsranks; r++) {
    ierr = ISRestoreIndices(sis[r], &sarr[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(rsize, rarr);CHKERRQ(ierr);
  ierr = PetscFree4(ssize, sarr, rreq, sreq);CHKERRQ(ierr);
  *ris = ris_;
  PetscFunctionReturn(0);
}

/* TODO VecExchangeBegin/End */
static PetscErrorCode ExchangeVecByRank_Private(PetscObject obj, PetscInt nsranks, const PetscMPIInt sranks[], Vec svecs[], PetscInt nrranks, const PetscMPIInt rranks[], Vec *rvecs[])
{
  PetscInt r;
  PetscInt *ssize, *rsize;
  PetscScalar **rarr;
  const PetscScalar **sarr;
  Vec *rvecs_;
  MPI_Request *sreq, *rreq;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc4(nsranks, &ssize, nsranks, &sarr, nrranks, &rreq, nsranks, &sreq);CHKERRQ(ierr);
  for (r=0; r<nsranks; r++) {
    ierr = VecGetLocalSize(svecs[r], &ssize[r]);CHKERRQ(ierr);
    ierr = VecGetArrayRead(svecs[r], &sarr[r]);CHKERRQ(ierr);
  }
  ierr = ExchangeArrayByRank_Private(obj, MPIU_SCALAR, nsranks, sranks, ssize, (const void**)sarr, nrranks, rranks, &rsize, (void***)&rarr);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrranks, &rvecs_);CHKERRQ(ierr);
  for (r=0; r<nrranks; r++) {
    /* set array in two steps to mimic PETSC_OWN_POINTER */
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, rsize[r], NULL, &rvecs_[r]);CHKERRQ(ierr);
    ierr = VecReplaceArray(rvecs_[r], rarr[r]);CHKERRQ(ierr);
  }
  for (r=0; r<nsranks; r++) {
    ierr = VecRestoreArrayRead(svecs[r], &sarr[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(rsize, rarr);CHKERRQ(ierr);
  ierr = PetscFree4(ssize, sarr, rreq, sreq);CHKERRQ(ierr);
  *rvecs = rvecs_;
  PetscFunctionReturn(0);
}

/* TODO add to DMPlex API */
PETSC_EXTERN PetscErrorCode DMPlexCheckConeOrientationOnInterfaces(DM);
PetscErrorCode DMPlexCheckConeOrientationOnInterfaces(DM dm)
{
  PetscSF             sf;
  PetscInt            nleaves, nranks, nroots;
  const PetscInt      *mine, *roffset, *rmine, *rremote;
  const PetscSFNode   *remote;
  const PetscMPIInt   *ranks;
  PetscSF             msf, imsf;
  PetscInt            nileaves, niranks;
  const PetscMPIInt   *iranks;
  const PetscInt      *iroffset, *irmine, *irremote;
  PetscInt            *rmine1, *rremote1; /* rmine and rremote copies simultaneously sorted by rank and rremote */
  PetscInt            *tmine, *mine_orig_numbering;
  const PetscInt      *degree;
  IS                  sntPointsPerRank, sntConesPerRank;
  Vec                 *sntCoordinatesPerRank;
  IS                  refPointsPerRank, refConesPerRank;
  Vec                 *refCoordinatesPerRank;
  Vec                 *recCoordinatesPerRank;
  PetscInt            i, j, k, n, o, r;
  PetscMPIInt         commsize, myrank;
  PetscBool           same;
  MPI_Comm            comm;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &myrank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &commsize);CHKERRQ(ierr);
  if (commsize < 2) PetscFunctionReturn(0);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  if (!sf) PetscFunctionReturn(0);
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &mine, &remote);CHKERRQ(ierr);
  if (nroots < 0) PetscFunctionReturn(0);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetRanks(sf, &nranks, &ranks, &roffset, &rmine, &rremote);CHKERRQ(ierr);

  {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] BEGIN DMPlexCheckConeOrientationOnInterfaces\n", myrank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
  }

  /* TODO: workaround for hang in DMGetCoordinates() (which is a bug since DMGetCoordinates() should be non-collective) */
  {
    Vec t;
    ierr = DMGetCoordinatesLocal(dm, &t);
  }

  /* Expand sent cones per rank */
  ierr = PetscMalloc2(nleaves, &rmine1, nleaves, &rremote1);CHKERRQ(ierr);
  for (r=0; r<nranks; r++) {
    /* simultaneously sort rank-wise portions of rmine & rremote by values in rremote
       - to unify order with the other side */
    o = roffset[r];
    n = roffset[r+1] - o;
    ierr = PetscMemcpy(&rmine1[o], &rmine[o], n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(&rremote1[o], &rremote[o], n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscSortIntWithArray(n, &rremote1[o], &rmine1[o]);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(nranks, &sntCoordinatesPerRank);CHKERRQ(ierr);
  for (r=0; r<nranks; r++) {
    o = roffset[r];
    n = roffset[r+1] - o;
    tmine = &rmine1[o];
    ierr = ISCreateGeneral(PETSC_COMM_SELF, n, tmine, PETSC_USE_POINTER, &sntPointsPerRank);CHKERRQ(ierr);
    ierr = DMPlexGetConesRecursive(dm, sntPointsPerRank, &sntConesPerRank);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinatesTuple(dm, sntConesPerRank, NULL, &sntCoordinatesPerRank[r]);CHKERRQ(ierr);
    ierr = ISDestroy(&sntPointsPerRank);CHKERRQ(ierr);
    ierr = ISDestroy(&sntConesPerRank);CHKERRQ(ierr);
  }

  /* Expand referenced cones per rank */
  /* - compute root degrees (nonzero indicates referenced point) */
  ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);

  /* - create inverse SF */
  ierr = PetscSFGetMultiSF(sf,&msf);CHKERRQ(ierr);
  ierr = PetscSFCreateInverseSF(msf,&imsf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(imsf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(imsf, NULL, &nileaves, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSFGetRanks(imsf, &niranks, &iranks, &iroffset, &irmine, &irremote);CHKERRQ(ierr);

  /* - compute original numbering of multi-roots (referenced points) */
  ierr = PetscMalloc1(nileaves, &mine_orig_numbering);CHKERRQ(ierr);
  for (i=0,j=0,k=0; i<nroots; i++) {
    if (!degree[i]) continue;
    for (j=0; j<degree[i]; j++,k++) {
      mine_orig_numbering[k] = i;
    }
  }
  if (PetscUnlikely(k != nileaves)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"sanity check fail");

  /* - expand cones per rank */
  ierr = PetscMalloc1(niranks, &refCoordinatesPerRank);CHKERRQ(ierr);
  for (r=0; r<niranks; r++) {
    o = iroffset[r];
    n = iroffset[r+1] - o;
    ierr = PetscMalloc1(n, &tmine);CHKERRQ(ierr);
    for (i=0; i<n; i++) tmine[i] = mine_orig_numbering[irmine[o+i]];
    ierr = ISCreateGeneral(PETSC_COMM_SELF, n, tmine, PETSC_OWN_POINTER, &refPointsPerRank);CHKERRQ(ierr);
    ierr = DMPlexGetConesRecursive(dm, refPointsPerRank, &refConesPerRank);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinatesTuple(dm, refConesPerRank, NULL, &refCoordinatesPerRank[r]);CHKERRQ(ierr);
    ierr = ISDestroy(&refPointsPerRank);CHKERRQ(ierr);
    ierr = ISDestroy(&refConesPerRank);CHKERRQ(ierr);
  }

  /* Send the coordinates */
  ierr = ExchangeVecByRank_Private((PetscObject)sf, nranks, ranks, sntCoordinatesPerRank, niranks, iranks, &recCoordinatesPerRank);CHKERRQ(ierr);

  /* Compare recCoordinatesPerRank with refCoordinatesPerRank */
  for (r=0; r<niranks; r++) {
    ierr = VecEqual(refCoordinatesPerRank[r], recCoordinatesPerRank[r], &same);CHKERRQ(ierr);
    if (!same) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "interface cones do not conform for remote rank %d", iranks[r]);
  }

  /* destroy sent stuff */
  for (r=0; r<nranks; r++) {
    ierr = VecDestroy(&sntCoordinatesPerRank[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree(sntCoordinatesPerRank);CHKERRQ(ierr);
  ierr = PetscFree2(rmine1, rremote1);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&imsf);CHKERRQ(ierr);

  /* destroy referenced stuff */
  for (r=0; r<niranks; r++) {
    ierr = VecDestroy(&refCoordinatesPerRank[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree(refCoordinatesPerRank);CHKERRQ(ierr);
  ierr = PetscFree(mine_orig_numbering);CHKERRQ(ierr);

  /* destroy received stuff */
  for (r=0; r<niranks; r++) {
    ierr = VecDestroy(&recCoordinatesPerRank[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree(recCoordinatesPerRank);CHKERRQ(ierr);

  {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] END DMPlexCheckConeOrientationOnInterfaces\n", myrank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexFixConeOrientationOnInterfaces_Private(DM dm)
{
  PetscSF             sf;
  PetscInt            nleaves, nranks, nroots;
  const PetscInt      *mine, *roffset, *rmine, *rremote;
  const PetscSFNode   *remote;
  const PetscMPIInt   *ranks;
  PetscSF             msf, imsf;
  PetscInt            nileaves, niranks;
  const PetscMPIInt   *iranks;
  const PetscInt      *iroffset, *irmine, *irremote;
  PetscInt            *rmine1, *rremote1; /* rmine and rremote copies simultaneously sorted by rank and rremote */
  PetscInt            *tmine, *tremote, *mine_orig_numbering;
  const PetscInt      *degree;
  IS                  *sntPointsPerRank, *sntConesPerRank, *sntConeRanksPerRank;
  PetscSection        *sntConesSectionPerRank;
  IS                  *refPointsPerRank, *refConesPerRank;
  PetscSection        *refConesSectionPerRank;
  IS                  *recConesPerRank, *recConeRanksPerRank;
  PetscInt            i, j, k, l, m, n, o, r;
  PetscMPIInt         commsize, myrank;
  MPI_Comm            comm;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &myrank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &commsize);CHKERRQ(ierr);
  if (commsize < 2) PetscFunctionReturn(0);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  if (!sf) PetscFunctionReturn(0);
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &mine, &remote);CHKERRQ(ierr);
  if (nroots < 0) PetscFunctionReturn(0);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetRanks(sf, &nranks, &ranks, &roffset, &rmine, &rremote);CHKERRQ(ierr);

  {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] BEGIN DMPlexFixConeOrientationOnInterfaces_Private\n", myrank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
    ierr = DMViewFromOptions(dm, NULL, "-before_fix_dm_view");CHKERRQ(ierr);
  }

  /* Expand sent cones per rank */
  ierr = PetscMalloc2(nleaves, &rmine1, nleaves, &rremote1);CHKERRQ(ierr);
  for (r=0; r<nranks; r++) {
    /* simultaneously sort rank-wise portions of rmine & rremote by values in rremote
       - to unify order with the other side */
    o = roffset[r];
    n = roffset[r+1] - o;
    ierr = PetscMemcpy(&rmine1[o], &rmine[o], n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(&rremote1[o], &rremote[o], n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscSortIntWithArray(n, &rremote1[o], &rmine1[o]);CHKERRQ(ierr);
  }
  ierr = PetscMalloc4(nranks, &sntPointsPerRank, nranks, &sntConesPerRank, nranks, &sntConesSectionPerRank, nranks, &sntConeRanksPerRank);CHKERRQ(ierr);
  for (r=0; r<nranks; r++) {
    o = roffset[r];
    n = roffset[r+1] - o;
    tmine = &rmine1[o];
    tremote = &rremote1[o];
    ierr = ISCreateGeneral(PETSC_COMM_SELF, n, tmine, PETSC_USE_POINTER, &sntPointsPerRank[r]);CHKERRQ(ierr);
    ierr = DMPlexGetConesTuple(dm, sntPointsPerRank[r], &sntConesSectionPerRank[r], &sntConesPerRank[r]);CHKERRQ(ierr);
    {
      AO ao;
      const PetscInt *sntConesPerRank_r_old;
      PetscInt *sntConesPerRank_r_, *sntConeRanksPerRank_r_;

      ierr = ISGetLocalSize(sntConesPerRank[r], &m);CHKERRQ(ierr);
      ierr = ISGetIndices(sntConesPerRank[r], &sntConesPerRank_r_old);CHKERRQ(ierr);
      ierr = PetscMalloc1(m, &sntConesPerRank_r_);CHKERRQ(ierr);
      ierr = PetscMalloc1(m, &sntConeRanksPerRank_r_);CHKERRQ(ierr);
      ierr = PetscMemcpy(sntConesPerRank_r_, sntConesPerRank_r_old, m*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = AOCreateMapping(PETSC_COMM_SELF, n, tmine, tremote, &ao);CHKERRQ(ierr);
      ierr = AOApplicationToPetsc(ao, m, sntConesPerRank_r_);CHKERRQ(ierr);
      for (i=0; i<m; i++) {
        if (sntConesPerRank_r_[i]<0) {
          /* point sntConesPerRank_r_old[i] not found in leaves for current rank ranks[r] -> find it among all leaves */
          ierr = PetscFindInt(sntConesPerRank_r_old[i], nleaves, mine, &l);CHKERRQ(ierr);
          sntConesPerRank_r_[i] = remote[l].index;
          sntConeRanksPerRank_r_[i] = (PetscInt) remote[l].rank;
        } else {
          sntConeRanksPerRank_r_[i] = (PetscInt) ranks[r];
        }
      }
      ierr = ISRestoreIndices(sntConesPerRank[r], &sntConesPerRank_r_old);CHKERRQ(ierr);
      ierr = ISGeneralSetIndices(sntConesPerRank[r], m, sntConesPerRank_r_, PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF, m, sntConeRanksPerRank_r_, PETSC_OWN_POINTER, &sntConeRanksPerRank[r]);CHKERRQ(ierr);
      ierr = AODestroy(&ao);CHKERRQ(ierr);
    }
  }

  /* Expand referenced cones per rank */
  /* - compute root degrees (nonzero indicates referenced point) */
  ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);

  /* - create inverse SF */
  ierr = PetscSFGetMultiSF(sf,&msf);CHKERRQ(ierr);
  ierr = PetscSFCreateInverseSF(msf,&imsf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(imsf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(imsf, NULL, &nileaves, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSFGetRanks(imsf, &niranks, &iranks, &iroffset, &irmine, &irremote);CHKERRQ(ierr);

  /* - compute original numbering of multi-roots (referenced points) */
  ierr = PetscMalloc1(nileaves, &mine_orig_numbering);CHKERRQ(ierr);
  for (i=0,j=0,k=0; i<nroots; i++) {
    if (!degree[i]) continue;
    for (j=0; j<degree[i]; j++,k++) {
      mine_orig_numbering[k] = i;
    }
  }
  if (PetscUnlikely(k != nileaves)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"sanity check fail");

  /* - expand cones per rank */
  ierr = PetscMalloc3(niranks, &refPointsPerRank, niranks, &refConesPerRank, niranks, &refConesSectionPerRank);CHKERRQ(ierr);
  for (r=0; r<niranks; r++) {
    o = iroffset[r];
    n = iroffset[r+1] - o;
    ierr = PetscMalloc1(n, &tmine);CHKERRQ(ierr);
    for (i=0; i<n; i++) tmine[i] = mine_orig_numbering[irmine[o+i]];
    ierr = ISCreateGeneral(PETSC_COMM_SELF, n, tmine, PETSC_OWN_POINTER, &refPointsPerRank[r]);CHKERRQ(ierr);
    ierr = DMPlexGetConesTuple(dm, refPointsPerRank[r], &refConesSectionPerRank[r], &refConesPerRank[r]);CHKERRQ(ierr);
  }

  /* send the cones */
  ierr = ExchangeISByRank_Private((PetscObject)sf, nranks, ranks, sntConesPerRank, niranks, iranks, &recConesPerRank);CHKERRQ(ierr);
  ierr = ExchangeISByRank_Private((PetscObject)sf, nranks, ranks, sntConeRanksPerRank, niranks, iranks, &recConeRanksPerRank);CHKERRQ(ierr);

  /* resolve non-local points in recConesPerRank */
  {
    PetscInt *cranks, *recc;
    PetscInt l, rr, size;

    for (r=0; r<niranks; r++) {
      ierr = ISGetLocalSize(recConeRanksPerRank[r], &size);CHKERRQ(ierr);
      /* we cheat because we know the is is general and that we can change the indices */
      ierr = ISGetIndices(recConeRanksPerRank[r], (const PetscInt**)&cranks);CHKERRQ(ierr);
      ierr = ISGetIndices(recConesPerRank[r], (const PetscInt**)&recc);CHKERRQ(ierr);
      for (i=0; i<size; i++) {
        if (((PetscMPIInt)cranks[i]) != myrank) {
          ierr = PetscFindMPIInt((PetscMPIInt)cranks[i], nranks, ranks, &rr);CHKERRQ(ierr);
          if (PetscUnlikely(rr<0)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "received rank %d not found among remote ranks",cranks[i]);
          o = roffset[rr];
          n = roffset[rr+1] - o;
          ierr = PetscFindInt(recc[i], n, &rremote1[o], &l);CHKERRQ(ierr);
          if (PetscUnlikely(l<0)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "point %d received from rank %d not found in remotes addressed to rank %d - provided SF is not proper point SF or the algorithm is not general enough",recc[i],iranks[r],cranks[i]);
          recc[i] = rmine1[l];
          cranks[i] = myrank;
        }
      }
      ierr = ISRestoreIndices(recConeRanksPerRank[r], (const PetscInt**)&cranks);CHKERRQ(ierr);
      ierr = ISRestoreIndices(recConesPerRank[r], (const PetscInt**)&recc);CHKERRQ(ierr);
    }
  }

  /* Compare recConesPerRank with refConesPerRank and adjust orientations */
  {
    PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);
  }
  for (r=0; r<niranks; r++) {
    {
      PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "[%d] r=%d iranks[r]=%d refConesPerRank[r] recConesPerRank[r]\n",myrank,r,iranks[r]);
      ISView(refConesPerRank[r], PETSC_VIEWER_STDOUT_SELF);
      ISView(recConesPerRank[r], PETSC_VIEWER_STDOUT_SELF);
    }
    ierr = DMPlexFixFaceOrientations_Private(dm, refPointsPerRank[r], refConesSectionPerRank[r], recConesPerRank[r], refConesPerRank[r]);CHKERRQ(ierr);
  }
  {
    PetscViewerFlush(PETSC_VIEWER_STDOUT_SELF);
    PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);
  }

  /* destroy sent stuff */
  for (r=0; r<nranks; r++) {
    ierr = ISDestroy(&sntPointsPerRank[r]);CHKERRQ(ierr);
    ierr = ISDestroy(&sntConesPerRank[r]);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&sntConesSectionPerRank[r]);CHKERRQ(ierr);
    ierr = ISDestroy(&sntConeRanksPerRank[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(sntPointsPerRank, sntConesPerRank, sntConesSectionPerRank, sntConeRanksPerRank);CHKERRQ(ierr);

  /* destroy referenced stuff */
  for (r=0; r<niranks; r++) {
    ierr = ISDestroy(&refPointsPerRank[r]);CHKERRQ(ierr);
    ierr = ISDestroy(&refConesPerRank[r]);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&refConesSectionPerRank[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree3(refPointsPerRank, refConesPerRank, refConesSectionPerRank);CHKERRQ(ierr);
  ierr = PetscFree(mine_orig_numbering);CHKERRQ(ierr);

  /* destroy received stuff */
  for (r=0; r<niranks; r++) {
    ierr = ISDestroy(&recConesPerRank[r]);CHKERRQ(ierr);
    ierr = ISDestroy(&recConeRanksPerRank[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree(recConesPerRank);CHKERRQ(ierr);
  ierr = PetscFree(recConeRanksPerRank);CHKERRQ(ierr);

  ierr = PetscFree2(rmine1, rremote1);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&imsf);CHKERRQ(ierr);
  {
    MPI_Barrier(PETSC_COMM_WORLD);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] END DMPlexFixConeOrientationOnInterfaces_Private\n", myrank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexInterpolate - Take in a cell-vertex mesh and return one with all intermediate faces, edges, etc.

  Collective on DM

  Input Parameters:
+ dm - The DMPlex object with only cells and vertices
- dmInt - The interpolated DM

  Output Parameter:
. dmInt - The complete DMPlex object

  Level: intermediate

  Notes:
    It does not copy over the coordinates.

.keywords: mesh
.seealso: DMPlexUninterpolate(), DMPlexCreateFromCellList(), DMPlexCopyCoordinates()
@*/
PetscErrorCode DMPlexInterpolate(DM dm, DM *dmInt)
{
  DM             idm, odm = dm;
  PetscSF        sfPoint;
  PetscInt       depth, dim, d;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmInt, 2);
  ierr = PetscLogEventBegin(DMPLEX_Interpolate,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if ((depth == dim) || (dim <= 1)) {
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    idm  = dm;
  } else {
    for (d = 1; d < dim; ++d) {
      /* Create interpolated mesh */
      ierr = DMCreate(PetscObjectComm((PetscObject)dm), &idm);CHKERRQ(ierr);
      ierr = DMSetType(idm, DMPLEX);CHKERRQ(ierr);
      ierr = DMSetDimension(idm, dim);CHKERRQ(ierr);
      if (depth > 0) {
        ierr = DMPlexInterpolateFaces_Internal(odm, 1, idm);CHKERRQ(ierr);
        ierr = DMGetPointSF(odm, &sfPoint);CHKERRQ(ierr);
        ierr = DMPlexInterpolatePointSF(idm, sfPoint, depth);CHKERRQ(ierr);
      }
      if (odm != dm) {ierr = DMDestroy(&odm);CHKERRQ(ierr);}
      odm = idm;
    }

    {
      PetscBool flg=PETSC_FALSE;
      PetscBool flg1=PETSC_FALSE;
      ierr = PetscOptionsGetBool(NULL, NULL, "-hotfix", &flg, NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetBool(NULL, NULL, "-hotfix1", &flg1, NULL);CHKERRQ(ierr);
      if (flg || flg1) {ierr = DMPlexHotfixInterpolatedPointSF_Private(idm);CHKERRQ(ierr);}
      if (flg) {ierr = PetscOptionsGetBool(NULL, NULL, "-cell_simplex", &flg, NULL);CHKERRQ(ierr);}
      if (flg) {
        PetscSF sf;
        PetscInt nroots,nleaves;
        const PetscInt *ilocal;
        const PetscSFNode *iremote;
        PetscInt *ilocal1;
        PetscSFNode *iremote1;
        PetscMPIInt rank;
        ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)idm),&rank);CHKERRQ(ierr);
        ierr = DMGetPointSF(idm, &sf);CHKERRQ(ierr);
        ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote);CHKERRQ(ierr);
        if (!rank) {
          ierr = PetscMalloc1(nleaves+1, &ilocal1);CHKERRQ(ierr);
          ierr = PetscMalloc1(nleaves+1, &iremote1);CHKERRQ(ierr);
          ierr = PetscMemcpy(ilocal1, ilocal, 3*sizeof(PetscInt));CHKERRQ(ierr);
          ierr = PetscMemcpy(iremote1, iremote, 3*sizeof(PetscSFNode));CHKERRQ(ierr);
          ilocal1[3] = 8;
          iremote1[3].rank = 1;
          iremote1[3].index = 6;
          ierr = PetscMemcpy(ilocal1+4, ilocal+3, 3*sizeof(PetscInt));CHKERRQ(ierr);
          ierr = PetscMemcpy(iremote1+4, iremote+3, 3*sizeof(PetscSFNode));CHKERRQ(ierr);
          ierr = PetscSFSetGraph(sf, nroots, nleaves+1, ilocal1, PETSC_OWN_POINTER, iremote1, PETSC_OWN_POINTER);CHKERRQ(ierr);
        } else {
          ierr = PetscSFSetGraph(sf, nroots, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
        }
      }
    }

    if (depth > 0) {
      PetscBool flg = PETSC_TRUE;
      ierr = PetscOptionsGetBool(NULL, NULL, "-dm_plex_fix_cone_orientation", &flg, NULL);CHKERRQ(ierr);
      if (flg) {ierr = DMPlexFixConeOrientationOnInterfaces_Private(idm);CHKERRQ(ierr);}
    }
    ierr = PetscObjectGetName((PetscObject) dm,  &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) idm,  name);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
    ierr = DMCopyLabels(dm, idm);CHKERRQ(ierr);
  }
  {
    PetscBool            isper;
    const PetscReal      *maxCell, *L;
    const DMBoundaryType *bd;

    ierr = DMGetPeriodicity(dm,&isper,&maxCell,&L,&bd);CHKERRQ(ierr);
    ierr = DMSetPeriodicity(idm,isper,maxCell,L,bd);CHKERRQ(ierr);
  }
  *dmInt = idm;
  ierr = PetscLogEventEnd(DMPLEX_Interpolate,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCopyCoordinates - Copy coordinates from one mesh to another with the same vertices

  Collective on DM

  Input Parameter:
. dmA - The DMPlex object with initial coordinates

  Output Parameter:
. dmB - The DMPlex object with copied coordinates

  Level: intermediate

  Note: This is typically used when adding pieces other than vertices to a mesh

.keywords: mesh
.seealso: DMCopyLabels(), DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM(), DMGetCoordinateSection()
@*/
PetscErrorCode DMPlexCopyCoordinates(DM dmA, DM dmB)
{
  Vec            coordinatesA, coordinatesB;
  VecType        vtype;
  PetscSection   coordSectionA, coordSectionB;
  PetscScalar   *coordsA, *coordsB;
  PetscInt       spaceDim, Nf, vStartA, vStartB, vEndA, vEndB, coordSizeB, v, d;
  PetscInt       cStartA, cEndA, cStartB, cEndB, cS, cE;
  PetscBool      lc = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  if (dmA == dmB) PetscFunctionReturn(0);
  ierr = DMPlexGetDepthStratum(dmA, 0, &vStartA, &vEndA);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dmB, 0, &vStartB, &vEndB);CHKERRQ(ierr);
  if ((vEndA-vStartA) != (vEndB-vStartB)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of vertices in first DM %d != %d in the second DM", vEndA-vStartA, vEndB-vStartB);
  ierr = DMPlexGetHeightStratum(dmA, 0, &cStartA, &cEndA);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmB, 0, &cStartB, &cEndB);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dmA, &coordSectionA);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dmB, &coordSectionB);CHKERRQ(ierr);
  if (coordSectionA == coordSectionB) PetscFunctionReturn(0);
  ierr = PetscSectionGetNumFields(coordSectionA, &Nf);CHKERRQ(ierr);
  if (!Nf) PetscFunctionReturn(0);
  if (Nf > 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of coordinate fields must be 1, not %D", Nf);
  if (!coordSectionB) {
    PetscInt dim;

    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) coordSectionA), &coordSectionB);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dmA, &dim);CHKERRQ(ierr);
    ierr = DMSetCoordinateSection(dmB, dim, coordSectionB);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject) coordSectionB);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetNumFields(coordSectionB, 1);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(coordSectionA, 0, &spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSectionB, 0, spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSectionA, &cS, &cE);CHKERRQ(ierr);
  if (cStartA <= cS && cS < cEndA) { /* localized coordinates */
    if ((cEndA-cStartA) != (cEndB-cStartB)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of cellls in first DM %d != %d in the second DM", cEndA-cStartA, cEndB-cStartB);
    cS = cS - cStartA + cStartB;
    cE = vEndB;
    lc = PETSC_TRUE;
  } else {
    cS = vStartB;
    cE = vEndB;
  }
  ierr = PetscSectionSetChart(coordSectionB, cS, cE);CHKERRQ(ierr);
  for (v = vStartB; v < vEndB; ++v) {
    ierr = PetscSectionSetDof(coordSectionB, v, spaceDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSectionB, v, 0, spaceDim);CHKERRQ(ierr);
  }
  if (lc) { /* localized coordinates */
    PetscInt c;

    for (c = cS-cStartB; c < cEndB-cStartB; c++) {
      PetscInt dof;

      ierr = PetscSectionGetDof(coordSectionA, c + cStartA, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(coordSectionB, c + cStartB, dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(coordSectionB, c + cStartB, 0, dof);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(coordSectionB);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSectionB, &coordSizeB);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmA, &coordinatesA);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinatesB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinatesB, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinatesB, coordSizeB, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinatesA, &d);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinatesB, d);CHKERRQ(ierr);
  ierr = VecGetType(coordinatesA, &vtype);CHKERRQ(ierr);
  ierr = VecSetType(coordinatesB, vtype);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesA, &coordsA);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesB, &coordsB);CHKERRQ(ierr);
  for (v = 0; v < vEndB-vStartB; ++v) {
    PetscInt offA, offB;

    ierr = PetscSectionGetOffset(coordSectionA, v + vStartA, &offA);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSectionB, v + vStartB, &offB);CHKERRQ(ierr);
    for (d = 0; d < spaceDim; ++d) {
      coordsB[offB+d] = coordsA[offA+d];
    }
  }
  if (lc) { /* localized coordinates */
    PetscInt c;

    for (c = cS-cStartB; c < cEndB-cStartB; c++) {
      PetscInt dof, offA, offB;

      ierr = PetscSectionGetOffset(coordSectionA, c + cStartA, &offA);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSectionB, c + cStartB, &offB);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(coordSectionA, c + cStartA, &dof);CHKERRQ(ierr);
      ierr = PetscMemcpy(coordsB + offB,coordsA + offA,dof*sizeof(*coordsB));CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(coordinatesA, &coordsA);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinatesB, &coordsB);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmB, coordinatesB);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinatesB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexUninterpolate - Take in a mesh with all intermediate faces, edges, etc. and return a cell-vertex mesh

  Collective on DM

  Input Parameter:
. dm - The complete DMPlex object

  Output Parameter:
. dmUnint - The DMPlex object with only cells and vertices

  Level: intermediate

  Notes:
    It does not copy over the coordinates.

.keywords: mesh
.seealso: DMPlexInterpolate(), DMPlexCreateFromCellList(), DMPlexCopyCoordinates()
@*/
PetscErrorCode DMPlexUninterpolate(DM dm, DM *dmUnint)
{
  DM             udm;
  PetscInt       dim, vStart, vEnd, cStart, cEnd, cMax, c, maxConeSize = 0, *cone;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmUnint, 2);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim <= 1) {
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    *dmUnint = dm;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &udm);CHKERRQ(ierr);
  ierr = DMSetType(udm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(udm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(udm, cStart, vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) ++coneSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(udm, c, coneSize);CHKERRQ(ierr);
    maxConeSize = PetscMax(maxConeSize, coneSize);
  }
  ierr = DMSetUp(udm);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxConeSize, &cone);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) cone[coneSize++] = p;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = DMPlexSetCone(udm, c, cone);CHKERRQ(ierr);
  }
  ierr = PetscFree(cone);CHKERRQ(ierr);
  ierr = DMPlexSetHybridBounds(udm, cMax, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(udm);CHKERRQ(ierr);
  ierr = DMPlexStratify(udm);CHKERRQ(ierr);
  /* Reduce SF */
  {
    PetscSF            sfPoint, sfPointUn;
    const PetscSFNode *remotePoints;
    const PetscInt    *localPoints;
    PetscSFNode       *remotePointsUn;
    PetscInt          *localPointsUn;
    PetscInt           vEnd, numRoots, numLeaves, l;
    PetscInt           numLeavesUn = 0, n = 0;
    PetscErrorCode     ierr;

    /* Get original SF information */
    ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
    ierr = DMGetPointSF(udm, &sfPointUn);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, NULL, &vEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
    /* Allocate space for cells and vertices */
    for (l = 0; l < numLeaves; ++l) if (localPoints[l] < vEnd) numLeavesUn++;
    /* Fill in leaves */
    if (vEnd >= 0) {
      ierr = PetscMalloc1(numLeavesUn, &remotePointsUn);CHKERRQ(ierr);
      ierr = PetscMalloc1(numLeavesUn, &localPointsUn);CHKERRQ(ierr);
      for (l = 0; l < numLeaves; l++) {
        if (localPoints[l] < vEnd) {
          localPointsUn[n]        = localPoints[l];
          remotePointsUn[n].rank  = remotePoints[l].rank;
          remotePointsUn[n].index = remotePoints[l].index;
          ++n;
        }
      }
      if (n != numLeavesUn) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent number of leaves %d != %d", n, numLeavesUn);
      ierr = PetscSFSetGraph(sfPointUn, vEnd, numLeavesUn, localPointsUn, PETSC_OWN_POINTER, remotePointsUn, PETSC_OWN_POINTER);CHKERRQ(ierr);
    }
  }
  {
    PetscBool            isper;
    const PetscReal      *maxCell, *L;
    const DMBoundaryType *bd;

    ierr = DMGetPeriodicity(dm,&isper,&maxCell,&L,&bd);CHKERRQ(ierr);
    ierr = DMSetPeriodicity(udm,isper,maxCell,L,bd);CHKERRQ(ierr);
  }

  *dmUnint = udm;
  PetscFunctionReturn(0);
}
