#define PETSCDM_DLL
#include <petsc-private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMPlexGmshRead"
PetscErrorCode DMPlexGmshRead(PetscViewer viewer, PetscBool byteSwap, void *data, PetscInt count, PetscDataType dtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dtype == PETSC_STRING) {
    PetscBool endl;
    PetscInt i = 0;
    char *buf = (char*)data;
    do {
      ierr = PetscViewerRead(viewer, &(buf[i]), 1, PETSC_CHAR);CHKERRQ(ierr);
      ierr = PetscStrncmp(&(buf[i]), "\n", 1, &endl);CHKERRQ(ierr);
      i++;
    } while (!endl);
    buf[i] = '\0';
  } else {
    ierr = PetscViewerRead(viewer, data, count, dtype);CHKERRQ(ierr);
  }
  if (byteSwap) {ierr = PetscByteSwap(data, dtype, count);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

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
  PetscViewerType vtype;
  GmshElement   *gmsh_elem;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords, *coordsIn = NULL;
  PetscInt       dim = 0, coordSize, c, v, d, cell, checkInt;
  int            numVertices = 0, numCells = 0, trueNumCells = 0;
  PetscMPIInt    num_proc, rank;
  char           endline, line[PETSC_MAX_PATH_LEN];
  PetscBool      match, binary, bswap = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &num_proc);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = PetscViewerGetType(viewer, &vtype);CHKERRQ(ierr);
  ierr = PetscStrcmp(vtype, PETSCVIEWERBINARY, &binary);CHKERRQ(ierr);
  if (!rank) {
    PetscBool match;
    int       fileType, dataSize, snum;

    /* Read header */
    ierr = DMPlexGmshRead(viewer, bswap, line, PETSC_MAX_PATH_LEN, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$MeshFormat\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = DMPlexGmshRead(viewer, bswap, line, PETSC_MAX_PATH_LEN, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "2.2 %d %d\n", &fileType, &dataSize);CHKERRQ(snum != 2);
    if (dataSize != sizeof(double)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data size %d is not valid for a Gmsh file", dataSize);
    if (binary) {
      ierr = DMPlexGmshRead(viewer, bswap, &checkInt, 1, PETSC_INT);CHKERRQ(ierr);
      if (checkInt != 1) {
        ierr = PetscByteSwap(&checkInt, PETSC_INT, 1);CHKERRQ(ierr);
        if (checkInt == 1) bswap = PETSC_TRUE;
        else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File type %d is not a valid Gmsh binary file", fileType);
      }
      ierr = DMPlexGmshRead(viewer, bswap, &endline, 1, PETSC_CHAR);CHKERRQ(ierr);
    } else {
      if (fileType) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File type %d is not a valid Gmsh ASCII file", fileType);
    }
    ierr = DMPlexGmshRead(viewer, bswap, line, PETSC_MAX_PATH_LEN, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$EndMeshFormat\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    /* Read vertices */
    ierr = DMPlexGmshRead(viewer, bswap, line, PETSC_MAX_PATH_LEN, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$Nodes\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = DMPlexGmshRead(viewer, bswap, line, PETSC_MAX_PATH_LEN, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d\n", &numVertices);CHKERRQ(snum != 1);
    ierr = PetscMalloc(numVertices*3 * sizeof(PetscScalar), &coordsIn);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {
      int    i;
      ierr = DMPlexGmshRead(viewer, bswap, &i, 1, PETSC_INT);CHKERRQ(ierr);
      ierr = DMPlexGmshRead(viewer, bswap, &(coordsIn[v*3]), 3, PETSC_DOUBLE);CHKERRQ(ierr);
      if (!binary) {ierr = DMPlexGmshRead(viewer, bswap, &endline, 1, PETSC_CHAR);CHKERRQ(ierr);}
      if (i != v+1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid node number %d should be %d", i, v+1);
    }
    if (binary) {ierr = DMPlexGmshRead(viewer, bswap, &endline, 1, PETSC_CHAR);CHKERRQ(ierr);}
    ierr = DMPlexGmshRead(viewer, bswap, line, PETSC_MAX_PATH_LEN, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$EndNodes\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    /* Read cells */
    ierr = DMPlexGmshRead(viewer, bswap, line, PETSC_MAX_PATH_LEN, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$Elements\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = DMPlexGmshRead(viewer, bswap, line, PETSC_MAX_PATH_LEN, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d\n", &numCells);CHKERRQ(snum != 1);
  }

  if (!rank) {
    /* Gmsh elements can be of any dimension/co-dimension, so we need to traverse the
       file contents multiple times to figure out the true number of cells and facets
       in the given mesh. To make this more efficient we read the file contents only
       once and store them in memory, while determining the true number of cells. */
    ierr = PetscMalloc1(numCells, &gmsh_elem);CHKERRQ(ierr);
    for (trueNumCells=0, c = 0; c < numCells; ++c) {
      ierr = DMPlexCreateGmsh_ReadElement(viewer, binary, bswap, &gmsh_elem[c]);CHKERRQ(ierr);
      if (gmsh_elem[c].dim > dim) {dim = gmsh_elem[c].dim; trueNumCells = 0;}
      if (gmsh_elem[c].dim == dim) trueNumCells++;
    }
    if (binary) {ierr = PetscViewerRead(viewer, &endline, 1, PETSC_CHAR);CHKERRQ(ierr);}
    ierr = DMPlexGmshRead(viewer, bswap, line, PETSC_MAX_PATH_LEN, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$EndElements\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
  }
  /* Allocate the cell-vertex mesh */
  ierr = DMPlexSetChart(*dm, 0, trueNumCells+numVertices);CHKERRQ(ierr);
  if (!rank) {
    for (cell = 0, c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim == dim) {
        ierr = DMPlexSetConeSize(*dm, cell, gmsh_elem[c].numNodes);CHKERRQ(ierr);
        cell++;
      }
    }
  }
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  /* Add cell-vertex connections */
  if (!rank) {
    PetscInt pcone[8], corner;
    for (cell = 0, c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim == dim) {
        for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
          pcone[corner] = gmsh_elem[c].nodes[corner] + trueNumCells-1;
        }
        ierr = DMPlexSetCone(*dm, cell, pcone);CHKERRQ(ierr);
        cell++;
      }
    }
  }
  ierr = MPI_Bcast(&dim, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  if (interpolate) {
    DM idm = NULL;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }

  if (!rank) {
    /* Apply boundary IDs by finding the relevant facets with vertex joins */
    PetscInt pcone[8], corner, vStart, vEnd;

    ierr = DMPlexGetDepthStratum(*dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    for (c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim == dim-1) {
        PetscInt joinSize;
        const PetscInt *join;
        for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
          pcone[corner] = gmsh_elem[c].nodes[corner] + vStart - 1;
        }
        ierr = DMPlexGetFullJoin(*dm, gmsh_elem[c].numNodes, (const PetscInt *) pcone, &joinSize, &join);CHKERRQ(ierr);
        if (joinSize != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for element %d", gmsh_elem[c].id);
        ierr = DMPlexSetLabelValue(*dm, "Face Sets", join[0], gmsh_elem[c].tags[0]);CHKERRQ(ierr);
        ierr = DMPlexRestoreJoin(*dm, gmsh_elem[c].numNodes, (const PetscInt *) pcone, &joinSize, &join);CHKERRQ(ierr);
      }
    }
  }

  /* Read coordinates */
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, trueNumCells, trueNumCells + numVertices);CHKERRQ(ierr);
  for (v = trueNumCells; v < trueNumCells+numVertices; ++v) {
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
  /* Clean up intermediate storage */
  if (!rank) {
    for (c = 0; c < numCells; ++c) {
      ierr = PetscFree(gmsh_elem[c].nodes);
      ierr = PetscFree(gmsh_elem[c].tags);
    }
    ierr = PetscFree(gmsh_elem);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateGmsh_ReadElement"
PetscErrorCode DMPlexCreateGmsh_ReadElement(PetscViewer viewer, PetscBool binary, PetscBool byteSwap, GmshElement *ele)
{
  int            cellType, numElem;
  char           endline;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (binary) {
    ierr = DMPlexGmshRead(viewer, byteSwap, &cellType, 1, PETSC_INT);CHKERRQ(ierr);
    ierr = DMPlexGmshRead(viewer, byteSwap, &numElem, 1, PETSC_INT);CHKERRQ(ierr);
    ierr = DMPlexGmshRead(viewer, byteSwap, &(ele->numTags), 1, PETSC_INT);CHKERRQ(ierr);
    ierr = DMPlexGmshRead(viewer, byteSwap, &(ele->id), 1, PETSC_INT);CHKERRQ(ierr);
  } else {
    ierr = DMPlexGmshRead(viewer, byteSwap, &(ele->id), 1, PETSC_INT);CHKERRQ(ierr);
    ierr = DMPlexGmshRead(viewer, byteSwap, &cellType, 1, PETSC_INT);CHKERRQ(ierr);
    ierr = DMPlexGmshRead(viewer, byteSwap, &(ele->numTags), 1, PETSC_INT);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(ele->numTags, &(ele->tags));CHKERRQ(ierr);
  ierr = DMPlexGmshRead(viewer, byteSwap, ele->tags, ele->numTags, PETSC_INT);CHKERRQ(ierr);
  switch (cellType) {
  case 1: /* 2-node line */
    ele->dim = 1;
    ele->numNodes = 2;
    break;
  case 2: /* 3-node triangle */
    ele->dim = 2;
    ele->numNodes = 3;
    break;
  case 3: /* 4-node quadrangle */
    ele->dim = 2;
    ele->numNodes = 4;
    break;
  case 4: /* 4-node tetrahedron */
    ele->dim  = 3;
    ele->numNodes = 4;
    break;
  case 5: /* 8-node hexahedron */
    ele->dim = 3;
    ele->numNodes = 8;
    break;
  case 15: /* 1-node vertex */
    ele->dim = 0;
    ele->numNodes = 1;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported Gmsh element type %d", cellType);
  }
  ierr = PetscMalloc1(ele->numNodes, &(ele->nodes));CHKERRQ(ierr);
  ierr = DMPlexGmshRead(viewer, byteSwap, ele->nodes, ele->numNodes, PETSC_INT);CHKERRQ(ierr);
  if (!binary) {ierr = PetscViewerRead(viewer, &endline, 1, PETSC_CHAR);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
