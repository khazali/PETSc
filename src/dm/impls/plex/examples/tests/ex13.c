static char help[] = "Build a mesh in parallel from a cell list and vertex mapping\n\n";

#include <petscdmplex.h>

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool numFacets;                    /* Number of facets in each dimension */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscInt  overlap;                      /* The cell overlap to use during partitioning */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim         = 2;
  options->numFacets   = 2;
  options->filename[0] = '\0';
  options->overlap     = 0;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex13.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-nfacets", "Number of facets in each dimension for unit meshes", "ex13.c", options->numFacets, &options->numFacets, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex13.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-overlap", "The cell overlap for partitioning", "ex13.c", options->overlap, &options->overlap, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             distMesh    = NULL;
  PetscInt       dim         = user->dim;
  const char    *filename    = user->filename;
  PetscInt       overlap     = user->overlap >= 0 ? user->overlap : 0;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    const char *extGmsh = ".msh";
    PetscBool   isGmsh;

    ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extGmsh, 4, &isGmsh);CHKERRQ(ierr);
    if (isGmsh) {
      PetscViewer viewer;

      ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
      ierr = DMPlexCreateGmsh(comm, viewer, PETSC_TRUE, dm);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      ierr = DMPlexCreateCGNSFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);
    }
  } else {
    DM boundary;
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};
    PetscInt  faces[3] = {user->numFacets, user->numFacets, user->numFacets};
    ierr = DMPlexCreate(comm, &boundary);CHKERRQ(ierr);
    ierr = DMSetType(boundary, DMPLEX);CHKERRQ(ierr);
    ierr = DMSetDimension(boundary, dim-1);CHKERRQ(ierr);
    if (dim == 2) {
      ierr = DMPlexCreateSquareBoundary(boundary, lower, upper, faces);CHKERRQ(ierr);
    } else {
      ierr = DMPlexCreateCubeBoundary(boundary, lower, upper, faces);CHKERRQ(ierr);
    }
    ierr = DMPlexGenerate(boundary, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMDestroy(&boundary);CHKERRQ(ierr);
  }
  ierr = DMPlexDistribute(*dm, overlap, NULL, &distMesh);CHKERRQ(ierr);
  if (distMesh) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = distMesh;
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Base Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "CreatePlexParallel"
PetscErrorCode CreatePlexParallel(DM dm, AppCtx *user)
{
  MPI_Comm       comm;
  PetscInt       dim = user->dim;
  DM             dmNew;
  PetscSF        pointSF, vertexSF;
  PetscInt       c, cStart, cEnd, vStart, vEnd, cl, clSize, idx;
  PetscInt      *cellList, *closure = NULL;
  Vec            coordsVec;
  PetscScalar   *coords;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
  ierr = PetscSFView(pointSF, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Extract cell-vertex list and coordinates */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1((dim+1)*(cEnd-cStart), &cellList);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; c++) {
    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
    for (idx = 0, cl = 0; cl < 2*clSize; cl+=2) {
      if (vStart <= closure[cl] && closure[cl] < vEnd) {
        cellList[c*(dim+1)+idx++] = closure[cl] - vStart;
      }
    }
  }
  ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coordsVec);CHKERRQ(ierr);
  ierr = VecGetArray(coordsVec, &coords);CHKERRQ(ierr);
  {
    PetscMPIInt        numProcs;
    PetscSFNode       *vremote;
    PetscInt          *vStartAll, *vlocal;
    PetscInt           l, nroots, nleaves;
    const PetscInt    *plocal;
    const PetscSFNode *premote;

    /* Build the SF that maps shared vertices */
    ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
    ierr = PetscMalloc1(numProcs, &vStartAll);CHKERRQ(ierr);
    ierr = MPI_Allgather(&vStart, 1, MPIU_INT, vStartAll, 1, MPIU_INT, comm);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(pointSF, &nroots, &nleaves, &plocal, &premote);CHKERRQ(ierr);
    ierr = PetscMalloc2(vEnd-vStart, &vlocal, vEnd-vStart, &vremote);CHKERRQ(ierr);
    for (idx = 0, l = 0; l < nleaves; l++) {
      if (vStart <= plocal[l] && plocal[l] < vEnd) {
        vlocal[idx] = plocal[l] - vStart;
        vremote[idx].index = premote[l].index - vStartAll[premote[l].rank];
        vremote[idx].rank = premote[l].rank;
        idx++;
      }
    }
    ierr = PetscSFCreate(comm, &vertexSF);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(vertexSF, vEnd-vStart, idx, vlocal, PETSC_OWN_POINTER, vremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscFree(vStartAll);CHKERRQ(ierr);
  }

  /* Re-build the DMPlex in parallel from cell-list, coordinates and vertex SF */
  ierr = DMPlexCreateFromCellList(comm, dim, cEnd-cStart, vEnd-vStart, dim+1, PETSC_TRUE, cellList, dim, coords, vertexSF, &dmNew);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmNew, NULL, "-dm_view");CHKERRQ(ierr);
  {
    PetscSF newPointSF;
    ierr = DMGetPointSF(dmNew, &newPointSF);CHKERRQ(ierr);
    ierr = PetscSFView(newPointSF, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordsVec, &coords);CHKERRQ(ierr);
  ierr = PetscFree(cellList);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&vertexSF);CHKERRQ(ierr);
  ierr = DMDestroy(&dmNew);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user; /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = CreatePlexParallel(dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
