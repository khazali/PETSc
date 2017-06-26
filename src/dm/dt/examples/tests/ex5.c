static char help[] = "Tests affine subspaces.\n\n";

#include <petscfe.h>

int main(int argc, char **argv)
{
  PetscSpace     space, subspace;
  PetscInt       Nv = 3 , Nc = 3, order = 3, subNv, subNc, i;
  PetscReal      *x, *Jx, *u, *Ju;
  PetscRandom    rand;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscSpaceCreate(PETSC_COMM_WORLD,&space);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(space,Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(space,Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetOrder(space,order);CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(space);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(space);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumVariables(space,&Nv);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(space,&Nc);CHKERRQ(ierr);
  subNv = PetscMax(1,Nv - 1);
  subNc = PetscMax(1,Nc - 1);
  ierr = PetscMalloc4(Nv,&x,Nv*subNv,&Jx,Nc,&u,Nc*subNc,&Ju);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  for (i = 0; i < Nv; i++) {ierr = PetscRandomGetValueReal(rand,&x[i]);CHKERRQ(ierr);}
  for (i = 0; i < Nv*subNv; i++) {ierr = PetscRandomGetValueReal(rand,&Jx[i]);CHKERRQ(ierr);}
  for (i = 0; i < Nc; i++) {ierr = PetscRandomGetValueReal(rand,&u[i]);CHKERRQ(ierr);}
  for (i = 0; i < Nc*subNc; i++) {ierr = PetscRandomGetValueReal(rand,&Ju[i]);CHKERRQ(ierr);}
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscSpaceCreateAffineSubspace(space,subNv,subNc,x,Jx,u,Ju,&subspace);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&subspace);CHKERRQ(ierr);
  ierr = PetscFree4(x,Jx,u,Ju);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&space);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
TEST*/
