static char help[] = "Test for hybrid meshing capabilities\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt      dim;                          /* The topological mesh dimension */
  PetscBool     interpolate;                  /* Generate intermediate mesh elements */
  char          meshfile[PETSC_MAX_PATH_LEN]; /* Input mesh file */
  char          hdf5file[PETSC_MAX_PATH_LEN]; /* Export mesh to file (HDF5) */
} AppCtx;


#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim               = 2;
  options->interpolate       = PETSC_FALSE;
  options->meshfile[0]       = '\0';
  options->hdf5file[0]       = '\0';

  ierr = PetscOptionsBegin(comm, "", "Hybrid Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex14.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-meshfile", "Output HDF5 file", "ex14.c", options->meshfile, options->meshfile, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-hdf5file", "Output HDF5 file", "ex14.c", options->hdf5file, options->hdf5file, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
};



/*
$      6
$     / \
$    / 1 \
$   /     \
$  5 ----- 4
$  |       |
$  |   0   |
$  |       |
$  2 ----- 3
*/
#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       len, dim        = user->dim;
  PetscBool      interpolate     = user->interpolate;
  const char    *filename        = user->meshfile;
  PetscMPIInt    rank, numProcs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);

  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len > 0) {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else {
    /* Create a 2D quad-tri mesh */
    ierr = DMPlexCreate(comm, dm);CHKERRQ(ierr);
    ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
    ierr = DMSetDimension(*dm, 2);CHKERRQ(ierr);
    if (dim == 2) {
      /* Non-interpolated hybrid quad-tri mesh */
      PetscInt    numPoints[2]         = {5, 2};
      PetscInt    coneSize[7]          = {4, 3, 0, 0, 0, 0, 0};
      PetscInt    cones[7]             = {2, 3, 4, 5,  4, 5, 6};
      PetscInt    coneOrientations[7]  = {0, 0, 0, 0, 0, 0, 0};
      PetscScalar vertexCoords[10]     = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 1.5};
      PetscInt    markerPoints[10]     = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};
      PetscInt    p;

      ierr = DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 5; ++p) {
        ierr = DMPlexSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      DM idm;
      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = DMPlexCopyCoordinates(*dm, idm);CHKERRQ(ierr);
      ierr = DMPlexCopyLabels(*dm, idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = idm;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Hybrid Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WriteMesh"
PetscErrorCode WriteMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscMPIInt    rank, numProcs;
  const char    *filename        = user->hdf5file;
  PetscViewer    viewer;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) PetscFunctionReturn(0);

  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERHDF5);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);

  /* Write DM */
  ierr = DMView(*dm, viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LoadMesh"
PetscErrorCode LoadMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscMPIInt    rank, numProcs;
  const char    *filename        = user->hdf5file;
  PetscViewer    viewer;
  size_t         len;
  PetscErrorCode ierr;

  ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMLoad(*dm, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) *dm, "Checkpoint Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm, cpdm;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = WriteMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = LoadMesh(PETSC_COMM_WORLD, &user, &cpdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&cpdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
