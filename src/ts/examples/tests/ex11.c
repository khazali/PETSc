static char help[] = "Tests TSTrajectoryGetVecs. \n\n";
/*
  This example tests TSTrajectory and the ability of TSTrajectoryGetVecs
  to reconstructs states and derivatives via interpolation (if necessary).
*/
#include <petscts.h>

PetscScalar func(PetscInt p, PetscReal t)  { return p ? t*func(p-1,t) : 1.0; }
PetscScalar dfunc(PetscInt p, PetscReal t)  { return p > 0 ? p*func(p-1,t) : 0.0; }

int main(int argc,char **argv)
{
  Vec            W,Wdot;
  TSTrajectory   tj;
  PetscReal      times[10];
  PetscReal      TT[10] = { 0.2, 0.9, 0.1, 0.3, 0.6, 0.7, 0.5, 1.0, 0.4, 0.8 };
  PetscInt       i, p = 1, Nt = 10;
  PetscInt       II[10] = { 1, 4, 9, 2, 3, 6, 5, 8, 10, 7 }; 
  PetscBool      sort;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = VecCreate(PETSC_COMM_WORLD,&W);CHKERRQ(ierr);
  ierr = VecSetSizes(W,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetUp(W);CHKERRQ(ierr);
  ierr = VecDuplicate(W,&Wdot);CHKERRQ(ierr);
  ierr = TSTrajectoryCreate(PETSC_COMM_WORLD,&tj);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(tj,NULL,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(tj,NULL);CHKERRQ(ierr);
  ierr = TSTrajectorySetUp(tj,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-interptimes",times,&Nt,NULL);CHKERRQ(ierr);
  sort = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-sortkeys",&sort,NULL);CHKERRQ(ierr);
  if (sort) {
    ierr = PetscSortReal(10,TT);CHKERRQ(ierr);
  }
  sort = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-sorttimes",&sort,NULL);CHKERRQ(ierr);
  if (sort) {
    ierr = PetscSortInt(10,II);CHKERRQ(ierr);
  }
  p = PetscMax(p,-p);
  for (i=0; i < 10; i++) {
    ierr = VecSet(W,func(p,TT[i]));CHKERRQ(ierr);
    ierr = TSTrajectorySet(tj,NULL,II[i],TT[i],W);CHKERRQ(ierr);
  }
  for (i = 0; i < Nt; i++) {
    PetscReal testtime = times[i];
    PetscScalar *aW,*aWdot;

    ierr = TSTrajectoryGetVecs(tj,NULL,PETSC_MIN_INT,&testtime,W,Wdot);CHKERRQ(ierr);
    ierr = VecGetArray(W,&aW);CHKERRQ(ierr);
    ierr = VecGetArray(Wdot,&aWdot);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0]));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0]));
    ierr = VecRestoreArray(W,&aW);CHKERRQ(ierr);
    ierr = VecRestoreArray(Wdot,&aWdot);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&W);CHKERRQ(ierr);
  ierr = VecDestroy(&Wdot);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&tj);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
