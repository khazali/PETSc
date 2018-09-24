#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>
#include <petsc/private/hashmapij.h>

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

static PetscErrorCode IntArrayViewFromOptions(MPI_Comm comm, const char opt[], const char name[], const char idxname[], const char valname[], PetscInt n, const PetscInt a[])
{
  PetscInt       idx;
  PetscMPIInt    rank;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL, NULL, opt, &flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(comm, "[%d]%s:\n", rank, name);CHKERRQ(ierr);
  for (idx = 0; idx < n; ++idx) {ierr = PetscSynchronizedPrintf(comm, "[%d]%s %D %s %D\n", rank, idxname, idx, valname, a[idx]);CHKERRQ(ierr);}
  ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SFNodeArrayViewFromOptions(MPI_Comm comm, const char opt[], const char name[], const char idxname[], PetscInt n, const PetscSFNode a[])
{
  PetscInt       idx;
  PetscMPIInt    rank;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL, NULL, opt, &flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(comm, "[%d]%s:\n", rank, name);CHKERRQ(ierr);
  if (idxname) {
    for (idx = 0; idx < n; ++idx) {ierr = PetscSynchronizedPrintf(comm, "[%d]%s %D rank %D index %D\n", rank, idxname, idx, a[idx].rank, a[idx].index);CHKERRQ(ierr);}
  } else {
    for (idx = 0; idx < n; ++idx) {ierr = PetscSynchronizedPrintf(comm, "[%d]rank %D index %D\n", rank, a[idx].rank, a[idx].index);CHKERRQ(ierr);}
  }
  ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMapToLocalPoint(PetscHMapIJ roothash, const PetscInt localPoints[], PetscMPIInt rank, PetscSFNode remotePoint, PetscInt *localPoint)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (remotePoint.rank == rank) {
    *localPoint = remotePoint.index;
  } else {
    PetscHashIJKey key;
    PetscInt       root;

    key.i = remotePoint.index;
    key.j = remotePoint.rank;
    ierr = PetscHMapIJGet(roothash, key, &root);CHKERRQ(ierr);
    if (root >= 0) {
      *localPoint = localPoints[root];
    } else PetscFunctionReturn(1);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexOrientPointSF_Internal(DM dm, PetscSF sf)
{
  PetscSFNode       *roots, *leaves;
  const PetscInt    *locals;
  const PetscSFNode *remotes;
  PetscInt           nroots, r, nleaves, l;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  PetscInt           debug = 0;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &locals, &remotes);CHKERRQ(ierr);
  if (nroots < 0) PetscFunctionReturn(0);
  /* If a point p is on the interface, then all its cone points must be also on interface  */
  ierr = PetscCalloc2(nroots, &roots, nroots, &leaves);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (debug && rank == 0) {ierr = PetscSynchronizedPrintf(comm, "Roots\n");CHKERRQ(ierr);}
  for (r = 0; r < nroots; ++r) {
    const PetscInt  point = r;
    const PetscInt *cone;
    PetscInt        coneSize, ind0, ind1;

    ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
    if (coneSize  > 1) {
      /* Translate all points to root numbering */
      ierr = PetscFindInt(cone[0], nleaves, locals, &ind0);CHKERRQ(ierr);
      ierr = PetscFindInt(cone[1], nleaves, locals, &ind1);CHKERRQ(ierr);
      roots[r].rank  = ind0 < 0 ? cone[0] : remotes[ind0].index;
      roots[r].index = ind1 < 0 ? cone[1] : remotes[ind1].index;
    }
    if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]  %D: %D %D\n", rank, r, roots[r].rank, roots[r].index);CHKERRQ(ierr);}
  }
  if (debug) {ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);}
  ierr = PetscSFBcastBegin(sf, MPIU_2INT, roots, leaves);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_2INT, roots, leaves);CHKERRQ(ierr);
  if (debug && rank == 0) {ierr = PetscSynchronizedPrintf(comm, "Leaves\n");CHKERRQ(ierr);}
  for (l = 0; l < nleaves; ++l) {
    const PetscInt  point = locals ? locals[l] : l;
    const PetscInt *cone;
    PetscInt        coneSize;

    if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]  %D: %D %D\n", rank, l, leaves[l].rank, leaves[l].index);CHKERRQ(ierr);}
    ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
    if (coneSize > 1) {
      PetscInt ind0, ind1;

      ierr = PetscFindInt(cone[0], nleaves, locals, &ind0);CHKERRQ(ierr);
      ierr = PetscFindInt(cone[1], nleaves, locals, &ind1);CHKERRQ(ierr);
      if (ind0 < 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point SF contains %D but is missing cone point %D = %D", point, 0, cone[0]);
      if (ind1 < 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point SF contains %D but is missing cone point %D = %D", point, 1, cone[1]);
      if ((leaves[point].rank  != remotes[ind0].index) || (leaves[point].index != remotes[ind1].index)) {
        const PetscInt *newcone;

        /* TODO Generalize this to correct an arbitrary orientation */
        ierr = DMPlexReverseCell(dm, point);CHKERRQ(ierr);
        if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]  reversed point %D\n", rank, point);CHKERRQ(ierr);}
        ierr = DMPlexGetCone(dm, point, &newcone);CHKERRQ(ierr);
        ierr = PetscFindInt(newcone[0], nleaves, locals, &ind0);CHKERRQ(ierr);
        ierr = PetscFindInt(newcone[1], nleaves, locals, &ind1);CHKERRQ(ierr);
        if (ind0 < 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point SF contains %D but is missing cone point %D = %D", point, 0, cone[0]);
        if (ind1 < 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point SF contains %D but is missing cone point %D = %D", point, 1, cone[1]);
        if (leaves[point].rank  != remotes[ind0].index)
          SETERRQ9(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D Cone[%D] %D --> (%D, %D) != %D Cone[%D] SF Point (%D, %D)", point, 0, cone[0], remotes[ind0].rank, remotes[ind0].index, leaves[point].rank,  0, remotes[l].rank, remotes[l].index);
        if (leaves[point].index != remotes[ind1].index)
          SETERRQ9(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D Cone[%D] %D --> (%D, %D) != %D Cone[%D] SF Point (%D, %D)", point, 1, cone[1], remotes[ind1].rank, remotes[ind1].index, leaves[point].index, 1, remotes[l].rank, remotes[l].index);
      }
    }
  }
  if (debug) {ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);}
  ierr = PetscFree2(roots, leaves);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexInterpolatePointSF - Insert interpolated points in the overlap into the PointSF in parallel, following local interpolation

  Collective on DM

  Input Parameters:
+ dm      - The interpolated DM
- pointSF - The initial SF without interpolated points

  Output Parameter:
. pointSF - The SF including interpolated points

  Level: intermediate

   Note: All debugging for this process can be turned on with the options: -dm_interp_pre_view -petscsf_interp_pre_view -petscsection_interp_candidate_view -petscsection_interp_candidate_remote_view -petscsection_interp_claim_view -petscsf_interp_pre_view -dmplex_interp_debug

.keywords: mesh
.seealso: DMPlexInterpolate(), DMPlexUninterpolate()
@*/
PetscErrorCode DMPlexInterpolatePointSF(DM dm, PetscSF pointSF)
{
  /*
       Okay, the algorithm is:
         - Take each point in the overlap (root)
         - Look at the neighboring points in the overlap (candidates)
         - Send these candidate points to neighbors
         - Neighbor checks for edge between root and candidate
         - If edge is found, it replaces candidate point with edge point
         - Send back the overwritten candidates (claims)
         - Original guy checks for edges, different from original candidate, and gets its own edge
         - This pair is put into SF

       We need a new algorithm that tolerates groups larger than 2.
         - Take each point in the overlap (root)
         - Find all collections of points in the overlap which make faces (do early join)
         - Send collections as candidates (add size as first number)
         - Neighbor check for face in collections
         - If face is found, it replaces candidate point with face point
         - Send back the overwritten candidates (claims)
         - Original guy checks for faces, different from original candidate, and gets its own face
         - This pair is put into SF
  */
  PetscHMapI         leafhash;
  PetscHMapIJ        roothash;
  const PetscInt    *localPoints, *rootdegree;
  const PetscSFNode *remotePoints;
  PetscSFNode       *candidates, *candidatesRemote, *claims;
  PetscSection       candidateSection, candidateSectionRemote, claimSection;
  PetscInt           numLeaves, l, numRoots, r, candidatesSize, candidatesRemoteSize;
  PetscMPIInt        size, rank;
  PetscHashIJKey     key;
  PetscBool          debug = PETSC_FALSE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL, ((PetscObject) dm)->prefix, "-dmplex_interp_debug", &debug);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(pointSF, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (size < 2 || numRoots < 0) PetscFunctionReturn(0);
  ierr = PetscObjectViewFromOptions((PetscObject) dm, NULL, "-dm_interp_pre_view");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) pointSF, NULL, "-petscsf_interp_pre_view");CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_InterpolateSF,dm,0,0,0);CHKERRQ(ierr);
  /* Build hashes of points in the SF for efficient lookup */
  ierr = PetscHMapICreate(&leafhash);CHKERRQ(ierr);
  ierr = PetscHMapIJCreate(&roothash);CHKERRQ(ierr);
  for (l = 0; l < numLeaves; ++l) {
    key.i = remotePoints[l].index;
    key.j = remotePoints[l].rank;
    ierr = PetscHMapISet(leafhash, localPoints[l], l);CHKERRQ(ierr);
    ierr = PetscHMapIJSet(roothash, key, l);CHKERRQ(ierr);
  }
  /* Compute root degree to identify sahred points */
  ierr = PetscSFComputeDegreeBegin(pointSF, &rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(pointSF, &rootdegree);CHKERRQ(ierr);
  ierr = IntArrayViewFromOptions(PetscObjectComm((PetscObject) dm), "-interp_root_degree_view", "Root degree", "point", "degree", numRoots, rootdegree);CHKERRQ(ierr);
  /* Build a section / SFNode array of candidate points (faces) in the cone(support(leaf)),
     where each candidate is defined by a set of remote points (roots) for the other vertices that define the face. */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &candidateSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(candidateSection, 0, numRoots);CHKERRQ(ierr);
  {
    PetscHMapI facehash;

    ierr = PetscHMapICreate(&facehash);CHKERRQ(ierr);
    for (l = 0; l < numLeaves; ++l) {
      const PetscInt    localPoint = localPoints[l];
      const PetscInt   *support;
      PetscInt          supportSize, s;

      if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Checking local point %D\n", rank, localPoint);CHKERRQ(ierr);}
      ierr = DMPlexGetSupportSize(dm, localPoint, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, localPoint, &support);CHKERRQ(ierr);
      for (s = 0; s < supportSize; ++s) {
        const PetscInt  face = support[s];
        const PetscInt *cone;
        PetscInt        coneSize, c, f, root;
        PetscBool       isFace = PETSC_TRUE;

        /* Only add face once */
        if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]    Support point %D\n", rank, face);CHKERRQ(ierr);}
        ierr = PetscHMapIGet(facehash, face, &f);CHKERRQ(ierr);
        if (f >= 0) continue;
        ierr = DMPlexGetConeSize(dm, face, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, face, &cone);CHKERRQ(ierr);
        /* If a cone point does not map to leaves on any proc, then do not put face in SF */
        for (c = 0; c < coneSize; ++c) {
          if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]      Cone point %D\n", rank, cone[c]);CHKERRQ(ierr);}
          ierr = PetscHMapIGet(leafhash, cone[c], &root);CHKERRQ(ierr);
          if (!rootdegree[cone[c]] && (root < 0)) {isFace = PETSC_FALSE; break;}
        }
        if (isFace) {
          if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]    Found face %D\n", rank, face);CHKERRQ(ierr);}
          ierr = PetscHMapISet(facehash, face, l);CHKERRQ(ierr);
          ierr = PetscSectionAddDof(candidateSection, localPoint, coneSize);CHKERRQ(ierr);
        }
      }
    }
    if (debug) {ierr = PetscSynchronizedFlush(PetscObjectComm((PetscObject) dm), NULL);CHKERRQ(ierr);}
    ierr = PetscHMapIClear(facehash);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(candidateSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(candidateSection, &candidatesSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(candidatesSize, &candidates);CHKERRQ(ierr);
    for (l = 0; l < numLeaves; ++l) {
      const PetscInt    localPoint = localPoints[l];
      const PetscInt   *support;
      PetscInt          supportSize, s, offset, idx = 0;

      if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Checking local point %D\n", rank, localPoint);CHKERRQ(ierr);}
      ierr = PetscSectionGetOffset(candidateSection, localPoint, &offset);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, localPoint, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, localPoint, &support);CHKERRQ(ierr);
      for (s = 0; s < supportSize; ++s) {
        const PetscInt  face = support[s];
        const PetscInt *cone;
        PetscInt        coneSize, c, f, root;
        PetscBool       isFace = PETSC_TRUE;

        /* Only add face once */
        if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]    Support point %D\n", rank, face);CHKERRQ(ierr);}
        ierr = PetscHMapIGet(facehash, face, &f);CHKERRQ(ierr);
        if (f >= 0) continue;
        ierr = DMPlexGetConeSize(dm, face, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, face, &cone);CHKERRQ(ierr);
        /* If a cone point does not map to leaves on any proc, then do not put face in SF */
        for (c = 0; c < coneSize; ++c) {
          if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]      Cone point %D\n", rank, cone[c]);CHKERRQ(ierr);}
          ierr = PetscHMapIGet(leafhash, cone[c], &root);CHKERRQ(ierr);
          if (!rootdegree[cone[c]] && (root < 0)) {isFace = PETSC_FALSE; break;}
        }
        if (isFace) {
          if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]    Adding face %D at idx %D\n", rank, face, idx);CHKERRQ(ierr);}
          ierr = PetscHMapISet(facehash, face, l);CHKERRQ(ierr);
          candidates[offset+idx].rank    = -1;
          candidates[offset+idx++].index = coneSize-1;
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == localPoint) continue;
            if (rootdegree[cone[c]]) {
              candidates[offset+idx].rank    = rank;
              candidates[offset+idx++].index = cone[c];
            } else {
              ierr = PetscHMapIGet(leafhash, cone[c], &root);CHKERRQ(ierr);
              if (root < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot locate local point %D in SF", cone[c]);
              candidates[offset+idx++] = remotePoints[root];
            }
          }
        }
      }
    }
    if (debug) {ierr = PetscSynchronizedFlush(PetscObjectComm((PetscObject) dm), NULL);CHKERRQ(ierr);}
    ierr = PetscHMapIDestroy(&facehash);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) candidateSection, NULL, "-petscsection_interp_candidate_view");CHKERRQ(ierr);
    ierr = SFNodeArrayViewFromOptions(PetscObjectComm((PetscObject) dm), "-petscsection_interp_candidate_view", "Candidates", NULL, candidatesSize, candidates);CHKERRQ(ierr);
  }
  /* Gather candidate section / array pair into the root partition via inverse(multi(pointSF)). */
  /*   Note that this section is indexed by offsets into leaves, not by point number */
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

    ierr = PetscObjectViewFromOptions((PetscObject) candidateSectionRemote, NULL, "-petscsection_interp_candidate_remote_view");CHKERRQ(ierr);
    ierr = SFNodeArrayViewFromOptions(PetscObjectComm((PetscObject) dm), "-petscsection_interp_candidate_remote_view", "Remote Candidates", NULL, candidatesRemoteSize, candidatesRemote);CHKERRQ(ierr);
  }
  /* */
  {
    PetscInt idx;
    /* There is a section point for every leaf attached to a given root point */
    for (r = 0, idx = 0; r < numRoots; ++r) {
      PetscInt deg;
      for (deg = 0; deg < rootdegree[r]; ++deg, ++idx) {
        PetscInt offset, dof, d;

        ierr = PetscSectionGetDof(candidateSectionRemote, idx, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(candidateSectionRemote, idx, &offset);CHKERRQ(ierr);
        for (d = 0; d < dof; ++d) {
          const PetscInt  sizeInd   = offset+d;
          const PetscInt  numPoints = candidatesRemote[sizeInd].index;
          const PetscInt *join      = NULL;
          PetscInt        points[1024], p, joinSize;

          points[0] = r;
          for (p = 0; p < numPoints; ++p) {
            ierr = DMPlexMapToLocalPoint(roothash, localPoints, rank, candidatesRemote[offset+(++d)], &points[p+1]);
            if (ierr) {d += numPoints-1 - p; break;} /* We got a point not in our overlap */
            if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Checking local candidate %D\n", rank, points[p+1]);CHKERRQ(ierr);}
          }
          if (ierr) continue;
          ierr = DMPlexGetJoin(dm, numPoints+1, points, &joinSize, &join);CHKERRQ(ierr);
          if (joinSize == 1) {
            if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]    Adding face %D at idx %D\n", rank, join[0], sizeInd);CHKERRQ(ierr);}
            candidatesRemote[sizeInd].rank  = rank;
            candidatesRemote[sizeInd].index = join[0];
          }
          ierr = DMPlexRestoreJoin(dm, numPoints+1, points, &joinSize, &join);CHKERRQ(ierr);
        }
      }
    }
    if (debug) {ierr = PetscSynchronizedFlush(PetscObjectComm((PetscObject) dm), NULL);CHKERRQ(ierr);}
  }
  /* Push claims back to receiver via the MultiSF and derive new pointSF mapping on receiver */
  {
    PetscSF         sfMulti, sfClaims, sfPointNew;
    PetscSFNode    *remotePointsNew;
    PetscHMapI      claimshash;
    PetscInt       *remoteOffsets, *localPointsNew;
    PetscInt        claimsSize, pStart, pEnd, root, numLocalNew, p, d;

    ierr = PetscSFGetMultiSF(pointSF, &sfMulti);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &claimSection);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfMulti, candidateSectionRemote, &remoteOffsets, claimSection);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(sfMulti, candidateSectionRemote, remoteOffsets, claimSection, &sfClaims);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(claimSection, &claimsSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(claimsSize, &claims);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sfClaims, MPIU_2INT, candidatesRemote, claims);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfClaims, MPIU_2INT, candidatesRemote, claims);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfClaims);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) claimSection, NULL, "-petscsection_interp_claim_view");CHKERRQ(ierr);
    ierr = SFNodeArrayViewFromOptions(PetscObjectComm((PetscObject) dm), "-petscsection_interp_claim_view", "Claims", NULL, claimsSize, claims);CHKERRQ(ierr);
    /* Walk the original section of local supports and add an SF entry for each updated item */
    ierr = PetscHMapICreate(&claimshash);CHKERRQ(ierr);
    for (p = 0; p < numRoots; ++p) {
      PetscInt dof, offset;

      if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Checking root for claims %D\n", rank, p);CHKERRQ(ierr);}
      ierr = PetscSectionGetDof(candidateSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(candidateSection, p, &offset);CHKERRQ(ierr);
      for (d = 0; d < dof;) {
        if (claims[offset+d].rank >= 0) {
          const PetscInt  faceInd   = offset+d;
          const PetscInt  numPoints = candidates[faceInd].index;
          const PetscInt *join      = NULL;
          PetscInt        joinSize, points[1024], c;

          if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]    Found claim for remote point (%D, %D)\n", rank, claims[faceInd].rank, claims[faceInd].index);CHKERRQ(ierr);}
          points[0] = p;
          if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]      point %D\n", rank, points[0]);CHKERRQ(ierr);}
          for (c = 0, ++d; c < numPoints; ++c, ++d) {
            key.i = candidates[offset+d].index;
            key.j = candidates[offset+d].rank;
            ierr = PetscHMapIJGet(roothash, key, &root);CHKERRQ(ierr);
            points[c+1] = localPoints[root];
            if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]      point %D\n", rank, points[c+1]);CHKERRQ(ierr);}
          }
          ierr = DMPlexGetJoin(dm, numPoints+1, points, &joinSize, &join);CHKERRQ(ierr);
          if (joinSize == 1) {
            if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]    Found local face %D\n", rank, join[0]);CHKERRQ(ierr);}
            ierr = PetscHMapISet(claimshash, join[0], faceInd);CHKERRQ(ierr);
          }
          ierr = DMPlexRestoreJoin(dm, numPoints+1, points, &joinSize, &join);CHKERRQ(ierr);
        } else d += claims[offset+d].index+1;
      }
    }
    if (debug) {ierr = PetscSynchronizedFlush(PetscObjectComm((PetscObject) dm), NULL);CHKERRQ(ierr);}
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
      PetscInt offset;
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
        ierr = DMPlexInterpolatePointSF(idm, sfPoint);CHKERRQ(ierr);
      }
      if (odm != dm) {ierr = DMDestroy(&odm);CHKERRQ(ierr);}
      odm = idm;
    }
    ierr = PetscObjectGetName((PetscObject) dm,  &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) idm,  name);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
    ierr = DMCopyLabels(dm, idm);CHKERRQ(ierr);
    ierr = DMGetPointSF(idm, &sfPoint);CHKERRQ(ierr);
    ierr = DMPlexOrientPointSF_Internal(idm, sfPoint);CHKERRQ(ierr);
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
