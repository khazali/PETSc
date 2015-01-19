#define PETSCDM_DLL
#include <petsc-private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateTriangleFromFile"
/*@C
  DMPlexCreateTriangleFromFile - Create a DMPlex mesh from a set of triangle files.

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
. basename - Base name of the set of triangle files
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Level: beginner

.keywords: mesh, Triangle
@*/
PetscErrorCode DMPlexCreateTriangleFromFile(MPI_Comm comm, const char basename[], PetscBool interpolate, DM *dm)
{
  size_t         len;
  const char    *extNode = ".node";
  const char    *extEle  = ".ele";
  const char    *extEdge = ".edge";
  const char    *extFace = ".face";
  char           fnameNode[PETSC_MAX_PATH_LEN], fnameEle[PETSC_MAX_PATH_LEN];
  char           fnameEdge[PETSC_MAX_PATH_LEN], fnameFace[PETSC_MAX_PATH_LEN];
  PetscViewer    vwrNode, vwrEle, vwrFacet = NULL;
  PetscBool      exists;
  PetscInt       dim = 2;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Create viewers for .node and .ele files */
  ierr = PetscStrlen(basename, &len);CHKERRQ(ierr);
  ierr = PetscStrcpy(fnameNode, basename);CHKERRQ(ierr);
  ierr = PetscStrcat(fnameNode, extNode);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &vwrNode);CHKERRQ(ierr);
  ierr = PetscViewerSetType(vwrNode, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(vwrNode, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(vwrNode, fnameNode);CHKERRQ(ierr);

  ierr = PetscStrcpy(fnameEdge, basename);CHKERRQ(ierr);
  ierr = PetscStrcat(fnameEdge, extEdge);CHKERRQ(ierr);
  ierr = PetscTestFile(fnameEdge, 'r', &exists);CHKERRQ(ierr);
  if (exists) {
    ierr = PetscViewerCreate(comm, &vwrFacet);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vwrFacet, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(vwrFacet, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vwrFacet, fnameEdge);CHKERRQ(ierr);
  }

  ierr = PetscStrcpy(fnameFace, basename);CHKERRQ(ierr);
  ierr = PetscStrcat(fnameFace, extFace);CHKERRQ(ierr);
  ierr = PetscTestFile(fnameFace, 'r', &exists);CHKERRQ(ierr);
  if (exists) {
    dim = 3;
    ierr = PetscViewerCreate(comm, &vwrFacet);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vwrFacet, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(vwrFacet, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vwrFacet, fnameFace);CHKERRQ(ierr);
  }

  ierr = PetscStrcpy(fnameEle, basename);CHKERRQ(ierr);
  ierr = PetscStrcat(fnameEle, extEle);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &vwrEle);CHKERRQ(ierr);
  ierr = PetscViewerSetType(vwrEle, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(vwrEle, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(vwrEle, fnameEle);CHKERRQ(ierr);

  ierr = DMPlexCreateTriangle(comm, dim, vwrNode, vwrEle, vwrFacet, interpolate, dm);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&vwrNode);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vwrEle);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vwrFacet);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateTriangle"
/*@C
  DMPlexCreateTriangle - Create a DMPlex mesh from a set of triangle file viewers.

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
. basename - Base name of the set of triangle files
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.cs.cmu.edu/~quake/triangle.html

  Level: beginner

.keywords: mesh, Triangle
@*/
PetscErrorCode DMPlexCreateTriangle(MPI_Comm comm, PetscInt dim, PetscViewer vwrNode, PetscViewer vwrEle, PetscViewer vwrFacet, PetscBool interpolate, DM *dm)
{
  PetscMPIInt    rank;
  char           line[PETSC_MAX_PATH_LEN];
  int            snum, numCells, numCellVertices, numVertices, vertexDim, numAttr, numBid;
  PetscInt       c, v, i, cell, vertex;
  PetscInt      *cellList, *attrBuffer, *bidBuffer;
  PetscScalar   *coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  /* Read cell list from .ele file */
  if (!rank) {
    do {ierr = PetscViewerReadLine(vwrEle, line);CHKERRQ(ierr);}
    while (line[0] == '#');
    snum = sscanf(line, "%d %d %d", &numCells, &numCellVertices, &numAttr);
    if (snum != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Triangle .ele file");
    ierr = PetscMalloc1(numCells*numCellVertices, &cellList);CHKERRQ(ierr);
    ierr = PetscMalloc1(numAttr, &attrBuffer);CHKERRQ(ierr);
    for (c = 0; c < numCells; c++) {
      ierr = PetscViewerRead(vwrEle, &cell, 1, PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerRead(vwrEle, &(cellList[c*numCellVertices]), numCellVertices, PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerRead(vwrEle, attrBuffer, numAttr, PETSC_INT);CHKERRQ(ierr);
      /* Correct Fortran cell numbering */
      for (i = 0; i < numCellVertices; i++) cellList[c*numCellVertices+i]--;
      if (c != cell-1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Triangle .ele file");
      /* Adjust vertex numbering for hex meshes */
      if (dim == 3 && numCellVertices == 8) {
        PetscInt mask[8] = {0, 1, 2, 3, 4, 7, 6, 5};
        ierr = PetscSortIntWithArray(8, mask, &(cellList[c*numCellVertices]));CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(attrBuffer);CHKERRQ(ierr);

    /* Read coordinates from .node file */
    do {ierr = PetscViewerReadLine(vwrNode, line);CHKERRQ(ierr);}
    while (line[0] == '#');
    snum = sscanf(line, "%d %d %d %d", &numVertices, &vertexDim, &numAttr, &numBid);
    if (snum != 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Triangle .node file");
    ierr = PetscMalloc1(numVertices*vertexDim, &coordinates);CHKERRQ(ierr);
    ierr = PetscMalloc2(numAttr, &attrBuffer, numBid, &bidBuffer);CHKERRQ(ierr);
    for (v = 0; v < numVertices; v++) {
      ierr = PetscViewerRead(vwrNode, &vertex, 1, PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerRead(vwrNode, &(coordinates[v*vertexDim]), vertexDim, PETSC_SCALAR);CHKERRQ(ierr);
      ierr = PetscViewerRead(vwrNode, attrBuffer, numAttr, PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerRead(vwrNode, bidBuffer, numBid, PETSC_INT);CHKERRQ(ierr);
    }
    ierr = PetscFree2(attrBuffer, bidBuffer);CHKERRQ(ierr);
  } else {
    numCells = 0; numVertices = 0;
  }
  ierr = MPI_Bcast(&numCellVertices, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&vertexDim, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);

  /* Create the mesh */
  ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCellVertices, interpolate, cellList, vertexDim, coordinates, dm);CHKERRQ(ierr);

  if (!rank && vwrFacet) {
    PetscInt        e, joinSize, numFacetVertices = -1;
    const PetscInt *join = NULL;
    int             numEdges, edge, vertices[3], bid;
    do {ierr = PetscViewerReadLine(vwrFacet, line);CHKERRQ(ierr);}
    while (line[0] == '#');
    snum = sscanf(line, "%d %d", &numEdges, &numBid);
    if (snum != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Triangle .edge file");
    /* Derive number of vertices per facet */
    if (dim == 2 && numCellVertices == 3) numFacetVertices = 2;
    else if (dim == 2 && numCellVertices == 4) numFacetVertices = 2;
    else if (dim == 3 && numCellVertices == 4) numFacetVertices = 3;
    else if (dim == 3 && numCellVertices == 8) numFacetVertices = 4;
    for (e = 0; e < numEdges; e++) {
      ierr = PetscViewerRead(vwrFacet, &edge, 1, PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerRead(vwrFacet, vertices, numFacetVertices, PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerRead(vwrFacet, &bid, 1, PETSC_INT);CHKERRQ(ierr);
      /* Correct Fortran vertex numbering */
      for (i = 0; i < numFacetVertices; i++) vertices[i] += numCells - 1;
      ierr = DMPlexGetFullJoin(*dm, numFacetVertices, vertices, &joinSize, &join);CHKERRQ(ierr);
      if (joinSize != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for edge %d", edge);
      ierr = DMPlexSetLabelValue(*dm, "Boundary Marker", join[0], bid);CHKERRQ(ierr);
      ierr = DMPlexRestoreJoin(*dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
    }
  }

  if (!rank) {ierr = PetscFree2(cellList, coordinates);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
