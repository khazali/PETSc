static char help[] = "Demonstration of creating and viewing DMFields objects.\n\n";

#include <petscdmfield.h>
#include <petscdmplex.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  DM             dm = NULL;
  MPI_Comm       comm;
  char           type[256] = DMPLEX;
  PetscBool      isda, isplex;
  PetscInt       dim = 2;
  DMField        field = NULL;
  PetscInt       nc = 1;
  PetscInt       numCells = -1;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMField Tutorial Options", "DM");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_type","DM implementation on which to define field","ex1.c",DMList,type,type,256,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","DM intrinsic dimension", "ex1.c", dim, &dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_components","Number of components in field", "ex1.c", nc, &nc, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (dim > 3) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"This examples works for dim <= 3, not %D",dim);
  ierr = PetscStrncmp(type,DMPLEX,256,&isplex);CHKERRQ(ierr);
  ierr = PetscStrncmp(type,DMDA,256,&isda);CHKERRQ(ierr);

  if (isplex) {
    PetscBool simplex     = PETSC_TRUE;
    PetscInt  cStart, cEnd, overlap = 0;

    ierr = PetscOptionsBegin(comm, "", "DMField DMPlex Options", "DM");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-simplex","Create a simplicial DMPlex","ex1.c",simplex,&simplex,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-overlap","DMPlex parallel overlap","ex1.c",overlap,&overlap,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (simplex) {
      PetscBool interpolate = PETSC_TRUE;
      PetscInt  numFaces    = 3;

      ierr = PetscOptionsBegin(comm, "", "DMField DMPlex Simplicial Options", "DM");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-num_faces","Number of edges per direction","ex1.c",numFaces,&numFaces,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-interpolate","Interpolate the DMPlex","ex1.c",interpolate,&interpolate,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
      ierr = DMPlexCreateBoxMesh(comm,dim,numFaces,interpolate,&dm);CHKERRQ(ierr);
    } else {
      PetscInt       cells[3] = {3,3,3};
      PetscInt       n = 3, i;
      PetscBool      flags[3];
      PetscInt       inttypes[3];
      DMBoundaryType types[3] = {DM_BOUNDARY_NONE};

      ierr = PetscOptionsBegin(comm, "", "DMField DMPlex Tensor Options", "DM");CHKERRQ(ierr);
      ierr = PetscOptionsIntArray("-cells","Cells per dimension","ex1.c",cells,&n,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEList("-boundary_x","type of boundary in x direction","ex1.c",DMBoundaryTypes,5,DMBoundaryTypes[types[0]],&inttypes[0],&flags[0]);CHKERRQ(ierr);
      ierr = PetscOptionsEList("-boundary_y","type of boundary in y direction","ex1.c",DMBoundaryTypes,5,DMBoundaryTypes[types[1]],&inttypes[1],&flags[1]);CHKERRQ(ierr);
      ierr = PetscOptionsEList("-boundary_z","type of boundary in z direction","ex1.c",DMBoundaryTypes,5,DMBoundaryTypes[types[2]],&inttypes[2],&flags[2]);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);

      for (i = 0; i < 3; i++) {
        if (flags[i]) {types[i] = (DMBoundaryType) inttypes[i];}
      }
      ierr = DMPlexCreateHexBoxMesh(comm,dim,cells,types[0],types[1],types[2],&dm);CHKERRQ(ierr);
    }
    {
      PetscPartitioner part;
      DM               dmDist;

      ierr = DMPlexGetPartitioner(dm,&part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetType(part,PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
      ierr = DMPlexDistribute(dm,overlap,NULL,&dmDist);CHKERRQ(ierr);
      if (dmDist) {
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm = dmDist;
      }
    }
    ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    numCells = cEnd - cStart;
  } else if (isda) {
    PetscInt       cStart, cEnd, i;
    PetscRandom    rand;
    PetscScalar    *cv;

    switch (dim) {
    case 1:
      ierr = DMDACreate1d(comm, DM_BOUNDARY_NONE, 3, 1, 1, NULL, &dm);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, &dm);CHKERRQ(ierr);
      break;
    default:
      ierr = DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 3, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, NULL, &dm);CHKERRQ(ierr);
      break;
    }
    ierr = DMDAGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    numCells = cEnd - cStart;
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
    ierr = PetscMalloc1(nc * (1 << dim),&cv);CHKERRQ(ierr);
    for (i = 0; i < nc * (1 << dim); i++) {
      PetscReal rv;

      ierr = PetscRandomGetValueReal(rand,&rv);CHKERRQ(ierr);
      cv[i] = rv;
    }
    ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
    ierr = DMFieldCreateDA(dm,nc,cv,&field);CHKERRQ(ierr);
    ierr = PetscFree(cv);CHKERRQ(ierr);
  } else SETERRQ1(comm,PETSC_ERR_SUP,"This test does not run for DM type %s",type);

  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)field,NULL,"-dmfield_view");CHKERRQ(ierr);
  ierr = DMFieldDestroy(&field);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

