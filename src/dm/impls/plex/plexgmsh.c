#define PETSCDM_DLL
#include <petsc-private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateGmsh"
/*@
  DMPlexCreateGmsh - Create a DMPlex mesh from a Gmsh file.

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
. viewer - The Viewer associated with a Gmsh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.geuz.org/gmsh/doc/texinfo/#MSH-ASCII-file-format

  Level: beginner

.keywords: mesh,Gmsh
.seealso: DMPLEX, DMCreate()
@*/
PetscErrorCode DMPlexCreateGmsh(MPI_Comm comm, PetscViewer viewer, PetscBool interpolate, DM *dm)
{
  FILE          *fd;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords, *coordsIn = NULL;
  PetscInt       dim = 0, coordSize, c, v, d;
  int            numVertices = 0, numCells = 0, snum;
  long           fpos = 0;
  PetscMPIInt    num_proc, rank;
  char           line[PETSC_MAX_PATH_LEN];
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &num_proc);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  if (!rank) {
    PetscBool match;
    int       fileType, dataSize;

    ierr = PetscViewerASCIIGetPointer(viewer, &fd);CHKERRQ(ierr);
    /* Read header */
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$MeshFormat\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    snum = fscanf(fd, "2.2 %d %d\n", &fileType, &dataSize);CHKERRQ(snum != 2);
    if (fileType) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File type %d is not a valid Gmsh ASCII file", fileType);
    if (dataSize != sizeof(double)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data size %d is not valid for a Gmsh file", dataSize);
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$EndMeshFormat\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    /* Read vertices */
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$Nodes\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    snum = fscanf(fd, "%d\n", &numVertices);CHKERRQ(snum != 1);
    ierr = PetscMalloc(numVertices*3 * sizeof(PetscScalar), &coordsIn);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {
      double x, y, z;
      int    i;

      snum = fscanf(fd, "%d %lg %lg %lg\n", &i, &x, &y, &z);CHKERRQ(snum != 4);
      coordsIn[v*3+0] = x; coordsIn[v*3+1] = y; coordsIn[v*3+2] = z;
      if (i != v+1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid node number %d should be %d", i, v+1);
    }
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$EndNodes\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    /* Read cells */
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$Elements\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    snum = fscanf(fd, "%d\n", &numCells);CHKERRQ(snum != 1);
  }
  ierr = DMPlexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);
  if (!rank) {
    fpos = ftell(fd);
    for (c = 0; c < numCells; ++c) {
      PetscInt numCorners, t;
      int      cone[8], i, cellType, numTags, tag;

      snum = fscanf(fd, "%d %d %d", &i, &cellType, &numTags);CHKERRQ(snum != 3);
      if (numTags) for (t = 0; t < numTags; ++t) {snum = fscanf(fd, "%d", &tag);CHKERRQ(snum != 1);}
      switch (cellType) {
      case 1: /* 2-node line */
        dim = 1;
        numCorners = 2;
        snum = fscanf(fd, "%d %d\n", &cone[0], &cone[1]);CHKERRQ(snum != numCorners);
        break;
      case 2: /* 3-node triangle */
        dim = 2;
        numCorners = 3;
        snum = fscanf(fd, "%d %d %d\n", &cone[0], &cone[1], &cone[2]);CHKERRQ(snum != numCorners);
        break;
      case 3: /* 4-node quadrangle */
        dim = 2;
        numCorners = 4;
        snum = fscanf(fd, "%d %d %d %d\n", &cone[0], &cone[1], &cone[2], &cone[3]);CHKERRQ(snum != numCorners);
        break;
      case 4: /* 4-node tetrahedron */
        dim  = 3;
        numCorners = 4;
        snum = fscanf(fd, "%d %d %d %d\n", &cone[0], &cone[1], &cone[2], &cone[3]);CHKERRQ(snum != numCorners);
        break;
      case 5: /* 8-node hexahedron */
        dim = 3;
        numCorners = 8;
        snum = fscanf(fd, "%d %d %d %d %d %d %d %d\n", &cone[0], &cone[1], &cone[2], &cone[3], &cone[4], &cone[5], &cone[6], &cone[7]);CHKERRQ(snum != numCorners);
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported Gmsh element type %d", cellType);
      }
      ierr = DMPlexSetConeSize(*dm, c, numCorners);CHKERRQ(ierr);
      if (i != c+1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell number %d should be %d", i, c+1);
    }
  }
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  if (!rank) {
    ierr = fseek(fd, fpos, SEEK_SET);CHKERRQ(ierr);
    for (c = 0; c < numCells; ++c) {
      PetscInt pcone[8], numCorners, corner, t;
      int      cone[8], i, cellType, numTags, tag;

      snum = fscanf(fd, "%d %d %d", &i, &cellType, &numTags);CHKERRQ(snum != 3);
      if (numTags) for (t = 0; t < numTags; ++t) {snum = fscanf(fd, "%d", &tag);CHKERRQ(snum != 1);}
      switch (cellType) {
      case 1: /* 2-node line */
        dim = 1;
        numCorners = 2;
        snum = fscanf(fd, "%d %d\n", &cone[0], &cone[1]);CHKERRQ(snum != numCorners);
        break;
      case 2: /* 3-node triangle */
        dim = 2;
        numCorners = 3;
        snum = fscanf(fd, "%d %d %d\n", &cone[0], &cone[1], &cone[2]);CHKERRQ(snum != numCorners);
        break;
      case 3: /* 4-node quadrangle */
        dim = 2;
        numCorners = 4;
        snum = fscanf(fd, "%d %d %d %d\n", &cone[0], &cone[1], &cone[2], &cone[3]);CHKERRQ(snum != numCorners);
        break;
      case 4: /* 4-node tetrahedron */
        dim  = 3;
        numCorners = 4;
        snum = fscanf(fd, "%d %d %d %d\n", &cone[0], &cone[1], &cone[2], &cone[3]);CHKERRQ(snum != numCorners);
        ierr = DMPlexInvertCell(dim, numCorners, cone);CHKERRQ(ierr);
        break;
      case 5: /* 8-node hexahedron */
        dim = 3;
        numCorners = 8;
        snum = fscanf(fd, "%d %d %d %d %d %d %d %d\n", &cone[0], &cone[1], &cone[2], &cone[3], &cone[4], &cone[5], &cone[6], &cone[7]);CHKERRQ(snum != numCorners);
        ierr = DMPlexInvertCell(dim, numCorners, cone);CHKERRQ(ierr);
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported Gmsh element type %d", cellType);
      }
      for (corner = 0; corner < numCorners; ++corner) pcone[corner] = cone[corner] + numCells-1;
      ierr = DMPlexSetCone(*dm, c, pcone);CHKERRQ(ierr);
      if (i != c+1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell number %d should be %d", i, c+1);
    }
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$EndElements\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
  }
  ierr = MPI_Bcast(&dim, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  if (interpolate) {
    DM idm;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }
  /* Read coordinates */
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
  for (v = numCells; v < numCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(comm, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(coordinates, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (!rank) {
    for (v = 0; v < numVertices; ++v) {
      for (d = 0; d < dim; ++d) {
        coords[v*dim+d] = coordsIn[v*3+d];
      }
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscFree(coordsIn);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
