static char help[] = "Test threadcomm with type=OpenMP,model=user with PETSc vector routines.\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec             x, y, a, b, c;
  PetscErrorCode  ierr;
  PetscInt        n=20;
  PetscScalar     alpha=3.0;
  MPI_Comm        comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Creating Vectors\n");CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,PETSC_DECIDE,n,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  PetscInt i, lsize;
    PetscScalar vnorm=0.0;

    // User gives threads to PETSc for threaded PETSc work

      ierr = VecSet(x,2.0);CHKERRCONTINUE(ierr);
      ierr = VecSet(y,3.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Test1 Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

    // Parallel threaded user code
    ierr = VecPointwiseMult(y,y,y);CHKERRCONTINUE(ierr);

    // User gives threads to PETSc for threaded PETSc work
      ierr = VecScale(y,2.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Test2 Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

    // User gives threads to PETSc for threaded PETSc work
      PetscScalar *avals, *bvals;
      ierr = PetscMalloc1(n,&avals);CHKERRCONTINUE(ierr);
      ierr = PetscMalloc1(n,&bvals);CHKERRCONTINUE(ierr);
      PetscInt pstart,pend;
      ierr = VecGetOwnershipRange(x,&pstart,&pend);CHKERRCONTINUE(ierr);
      lsize = pend - pstart;
      //printf("lsize=%d\n",lsize);
      for(i=pstart; i<pend; i++) {
        avals[i] = pstart + i + 1.0;
        bvals[i] = -n+5 + (pstart + i)*2.0;
      }
      ierr = VecCreateMPIWithArray(comm,PETSC_DECIDE,lsize,n,avals,&a);CHKERRCONTINUE(ierr);
      ierr = VecCreateMPIWithArray(comm,PETSC_DECIDE,lsize,n,bvals,&b);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(x,&c);CHKERRCONTINUE(ierr);
      //VecView(b,PETSC_VIEWER_STDOUT_WORLD);

      // Vec reductions
      PetscInt vminind, vmaxind;
      PetscScalar vmin, vmax;
      ierr = VecMin(a,&vminind,&vmin);CHKERRCONTINUE(ierr);
      ierr = VecMax(a,&vmaxind,&vmax);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Test3 Min=%f Max=%f\n",vmin,vmax);CHKERRCONTINUE(ierr);

      // Vec Pointwise
      ierr = VecPointwiseMult(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Test3 PMult Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecPointwiseDivide(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Test3 PDiv Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecPointwiseMax(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Test3 PMax Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecPointwiseMin(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Test3 PMin Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecPointwiseMaxAbs(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Test3 PMaxAbs Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      // Vec multiple dot product
      VecSet(c,5.0);
      Vec mvecs[2];
      PetscScalar vals[2];
      mvecs[0] = a;
      mvecs[1] = b;
      VecMDot(c,2,mvecs,vals);
      ierr = PetscPrintf(comm,"Test3 MDot n1=%f n2=%f\n",vals[0],vals[1]);CHKERRCONTINUE(ierr);
      ierr = PetscFree(avals);CHKERRCONTINUE(ierr);
      ierr = PetscFree(bvals);CHKERRCONTINUE(ierr);

  // Destroy Vecs
  ierr = PetscPrintf(comm,"Destory and Finalize\n");CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&c);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
