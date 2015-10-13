static char help[] = "Tests non-conforming mesh manipulation\n\n";

/*     11         20
  3----------4----------5
  |         / \         |
  |         | |21  2    |19
  |         | |         |
12|   0   14| 9---------10
  |         | |    17   |
  |         | |18  1    |16
  |         \ /         |
  6----------7----------8
       13         15
*/

#include <petscdmplex.h>

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim; /* The topological mesh dimension */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim = 2;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex13.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             dmDist = NULL;
  PetscInt       dim    = user->dim;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Create the simplest nonconforming hex mesh */
  switch (dim) {
  case 2:
    ierr = DMCreate(comm, dm);CHKERRQ(ierr);
    ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
    ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
    if (!rank) {
      ierr = DMPlexSetChart(*dm, 0, 25);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 0, 4);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 1, 4);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 2, 4);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 11, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 12, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 13, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 14, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 15, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 16, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 17, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 18, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 19, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 20, 2);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(*dm, 21, 2);CHKERRQ(ierr);
    }
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    if (!rank) {
      PetscInt cone[4];
      PetscInt ornt[4];

      cone[0] = 11; cone[1] = 12; cone[2] = 13; cone[3] = 14;
      ierr = DMPlexSetCone(*dm, 0, cone);CHKERRQ(ierr);
      cone[0] = 15; cone[1] = 16; cone[2] = 17; cone[3] = 18;
      ierr = DMPlexSetCone(*dm, 1, cone);CHKERRQ(ierr);
      cone[0] = 17; cone[1] = 19; cone[2] = 20; cone[3] = 21;
      ornt[0] = -2; ornt[1] = 0;  ornt[2] = 0;  ornt[3] = 0;
      ierr = DMPlexSetCone(*dm, 2, cone);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(*dm, 2, ornt);CHKERRQ(ierr);
      cone[0] = 4; cone[1] = 3;
      ierr = DMPlexSetCone(*dm, 11, cone);CHKERRQ(ierr);
      cone[0] = 3; cone[1] = 6;
      ierr = DMPlexSetCone(*dm, 12, cone);CHKERRQ(ierr);
      cone[0] = 6; cone[1] = 7;
      ierr = DMPlexSetCone(*dm, 13, cone);CHKERRQ(ierr);
      cone[0] = 7; cone[1] = 4;
      ierr = DMPlexSetCone(*dm, 14, cone);CHKERRQ(ierr);
      cone[0] = 7; cone[1] = 8;
      ierr = DMPlexSetCone(*dm, 15, cone);CHKERRQ(ierr);
      cone[0] = 8; cone[1] = 10;
      ierr = DMPlexSetCone(*dm, 16, cone);CHKERRQ(ierr);
      cone[0] = 10; cone[1] = 9;
      ierr = DMPlexSetCone(*dm, 17, cone);CHKERRQ(ierr);
      cone[0] = 9; cone[1] = 7;
      ierr = DMPlexSetCone(*dm, 18, cone);CHKERRQ(ierr);
      cone[0] = 10; cone[1] = 5;
      ierr = DMPlexSetCone(*dm, 19, cone);CHKERRQ(ierr);
      cone[0] = 5; cone[1] = 5;
      ierr = DMPlexSetCone(*dm, 20, cone);CHKERRQ(ierr);
      cone[0] = 4; cone[1] = 9;
      ierr = DMPlexSetCone(*dm, 21, cone);CHKERRQ(ierr);
    }
    ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No sample mesh for dimension %D", dim);
  }
  /* Create labels */
  ierr = DMPlexDistribute(*dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = dmDist;
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Nonconforming Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
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
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
