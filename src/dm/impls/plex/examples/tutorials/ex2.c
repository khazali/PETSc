static char help[] = "Read in a mesh and test whether it is valid\n\n";

#include <petscdmplex.h>
#if defined(PETSC_HAVE_CGNS)
#include <cgnslib.h>
#endif
#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#endif

typedef struct {
  PetscBool interpolate;                  /* Generate intermediate mesh elements */
  char      filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
  PetscErrorCode (**BCFuncs)(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
} AppCtx;

PETSC_STATIC_INLINE PetscErrorCode zero(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  int i;
  for (i = 0 ; i < dim ; i++) u[i] = 0.;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options, PetscInt *a_dim)
{
  PetscErrorCode ierr;
  PetscInt       dim = 2;
  PetscFunctionBeginUser;
  options->interpolate = PETSC_TRUE;
  options->filename[0] = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex2.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex2.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "dimension of problem (2 or 3), used for non-file mesh", "ex2.c", dim, &dim, NULL);CHKERRQ(ierr);
  *a_dim = dim;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, PetscInt dim, DM *dm)
{
  PetscErrorCode ierr;
  size_t         len;
  PetscFunctionBeginUser;
  ierr = PetscStrlen(user->filename, &len);CHKERRQ(ierr);
  if (!len) {
    PetscInt numFaces = 2, id;
    DMLabel  label;
    ierr = DMPlexCreateBoxMesh(comm, dim, numFaces, user->interpolate, dm);CHKERRQ(ierr);
    /* need to do BCs that should be in the file */
    ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, label);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(PetscErrorCode (*)(PetscInt,const PetscReal [],PetscInt,PetscScalar*,void*)),&user->BCFuncs);CHKERRQ(ierr);
    user->BCFuncs[0] = zero;
    id = 1;
    ierr = DMAddBoundary(*dm, PETSC_TRUE, "wall", "boundary", 0, 0, NULL, (void (*)()) user->BCFuncs[0], 1, &id, user);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->filename, user->interpolate, dm);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckMeshTopology"
static PetscErrorCode CheckMeshTopology(DM dm)
{
  PetscInt       dim, coneSize, cStart;
  PetscBool      isSimplex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
  isSimplex = coneSize == dim+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = DMPlexCheckSymmetry(dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSkeleton(dm, isSimplex, 0);CHKERRQ(ierr);
  ierr = DMPlexCheckFaces(dm, isSimplex, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckMeshGeometry"
static PetscErrorCode CheckMeshGeometry(DM dm)
{
  PetscInt       dim, coneSize, cStart, cEnd, c;
  PetscReal     *v0, *J, *invJ, detJ;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for cell %d", detJ, c);
  }
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;
  PetscInt       dim;
  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user, &dim);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, dim, &dm);CHKERRQ(ierr);
  ierr = CheckMeshTopology(dm);CHKERRQ(ierr);
  ierr = CheckMeshGeometry(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
