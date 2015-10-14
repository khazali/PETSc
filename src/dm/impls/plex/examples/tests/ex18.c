static char help[] = "Tests non-conforming mesh manipulation\n\n";


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
  DM             refTree = NULL;
  PetscInt       dim    = user->dim;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  /* Create the simplest nonconforming hex mesh */
  switch (dim) {
  case 2:
    if (!rank) {
      /*     11         20
        3<---------4<---------5
        |         / \         ^
        |         ^ ^21  2    |19
        |         | |         |
      12|   0   14| 9<--------10
        |         | ^    17   ^
        |         | |18  1    |16
        v         \ /         |
        6--------->7--------->8
             13         15
      */
      PetscInt numPoints[3] = {8, 11, 3}; /* vertices, edges, quads */
      PetscInt coneSize[22] = {4, 4, 4,                          /* quads */
                               0, 0, 0, 0, 0, 0, 0, 0,           /* vertices */
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}; /* edges */
      PetscInt cones[34] = {11, 12, 13, 14, /* 0 */
                            15, 16, 17, 18, /* 1 */
                            17, 19, 20, 21, /* 2 */
                            4,  3,          /* 11 */
                            3,  6,          /* 12 */
                            6,  7,          /* 13 */
                            7,  4,          /* 14 */
                            7,  8,          /* 15 */
                            8,  10,         /* 16 */
                            10, 9,          /* 17 */
                            7,  9,          /* 18 */
                            10, 5,          /* 19 */
                            5,  4,          /* 20 */
                            9,  4};         /* 21 */
      PetscInt ornts[34] = { 0,  0,  0,  0, /* edges of 0 */
                             0,  0,  0, -2, /* edges of 1: because 14 points from 7 to 4, 18 must point from 7 to 9 */
                            -2,  0,  0, -2, /* edges of 2: because 14 points from 7 to 4, 21 must point from 9 to 4 */
                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; /* vertices have no orientation */
      PetscScalar vertices[16] = {0.,2.,  /* 3 */
                                  2.,2.,  /* 4 */
                                  3.,2.,  /* 5 */
                                  0.,0.,  /* 6 */
                                  2.,0.,  /* 7 */
                                  3.,0.,  /* 8 */
                                  2.,1.,  /* 9 */
                                  3.,1.}; /* 10 */

      ierr = DMPlexCreateFromDAG(*dm,3,numPoints,coneSize,cones,ornts,vertices);CHKERRQ(ierr);
    }
    else {
      PetscInt numPoints[3] = {0, 0, 0};

      ierr = DMPlexCreateFromDAG(*dm,3,numPoints,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No sample mesh for dimension %D", dim);
  }
  /* Create the reference tree describing the non-conforming interfaces */
                                                  /* PETSC_FALSE = quad/hex */
  ierr = DMPlexCreateDefaultReferenceTree(comm, dim, PETSC_FALSE, &refTree);CHKERRQ(ierr);
  ierr = DMPlexSetReferenceTree(*dm,refTree);CHKERRQ(ierr);
  ierr = DMDestroy(&refTree);CHKERRQ(ierr);
  /* Create the hierarchy of points */
  switch (dim) {
  case 2:
    {
      PetscSection parentSection;
      PetscInt     *parents, *childIDs, pStart, pEnd;

      /* If we really wanted to we could chose a more restrictive chart */
      ierr = DMPlexGetChart(*dm,&pStart,&pEnd);CHKERRQ(ierr);
      ierr = PetscSectionCreate(comm,&parentSection);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(parentSection,pStart,pEnd);CHKERRQ(ierr);
      if (!rank) {

        /* Each of these points has a parent */
        ierr = PetscSectionSetDof(parentSection,9, 1);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(parentSection,18,1);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(parentSection,21,1);CHKERRQ(ierr);
        ierr = PetscMalloc2(3,&parents,3,&childIDs);CHKERRQ(ierr);
        /* The parent of each of these points is 14 */
        parents[0] = 14; /* 9  */
        parents[1] = 14; /* 18 */
        parents[2] = 14; /* 21 */
        /* Now we have to tell DMPlex which part of the reference tree the non-conforming mesh looks like.  This is the
         * least friendly part of setup */
        /* 9 is to its parent (14) as 25 is to its parent (5) in the reference tree */
        childIDs[0] = 25;
        /* 18 is to its parent (14) as 9 is to its parent (5) in the reference tree */
        childIDs[1] = 9;
        /* 21 is to its parent (14) as 10 is to its parent (5) in the reference tree */
        childIDs[1] = 10;
      }
      else {
        ierr = PetscMalloc2(0,&parents,0,&childIDs);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetUp(parentSection);CHKERRQ(ierr);
      ierr = DMPlexSetTree(*dm,parentSection,parents,childIDs);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&parentSection);CHKERRQ(ierr);
      ierr = PetscFree2(parents,childIDs);CHKERRQ(ierr);
    }
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
