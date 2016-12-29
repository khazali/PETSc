
static char help[] = "Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()\n";

#include <petscsys.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);
  ierr = PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  ierr = PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
