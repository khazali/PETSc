static char help[] = "Darcy flow problem in 2d and 3d, using\n\
SNES concepts for optimization and statistical inference.\n\n\n";

#include <petscsnes.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    args:

TEST*/
