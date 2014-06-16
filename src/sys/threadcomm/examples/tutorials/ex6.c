static char help[] = "Test PetscThreadPool with OpenMP with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <omp.h>
#include <petscthreadcomm.h>
#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec             x, y, a, b, c;
  PetscErrorCode  ierr;
  PetscInt        nthreads, n=20, lsize;
  PetscScalar     alpha=3.0;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscThreadPoolGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"nthreads=%d\n",nthreads);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Creating vecs\n");CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&c);CHKERRQ(ierr);

  #pragma omp parallel num_threads(nthreads) default(shared) private(ierr)
  {
    PetscInt i, prank, start, end, *indices;
    PetscScalar vnorm=0.0;
    PetscScalar *ay;
    int trank = omp_get_thread_num();

    // User gives threads to PETSc for threaded PETSc work
    ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"trank=%d joined pool prank=%d\n",trank,prank);
    if(prank>=0) {
      PetscPrintf(PETSC_COMM_WORLD,"Vec work\n");
      ierr = VecSet(x,2.0);CHKERRCONTINUE(ierr);
      ierr = VecSet(y,3.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);

    // Get data for local work
    ierr = VecGetArray(y,&ay);CHKERRCONTINUE(ierr);
    ierr = VecGetLocalSize(x,&lsize);CHKERRCONTINUE(ierr);
    ierr = PetscThreadCommGetOwnershipRanges(PETSC_COMM_WORLD,lsize,&indices);CHKERRCONTINUE(ierr);

    // Parallel threaded user code
    start = indices[trank];
    end = indices[trank+1];
    ierr = PetscPrintf(PETSC_COMM_WORLD,"trank=%d start=%d end=%d\n",trank,start,end);CHKERRCONTINUE(ierr);
    for(i=start; i<end; i++) {
      ay[i] = ay[i]*ay[i];
    }

    // Restore vector
    ierr = VecRestoreArray(y,&ay);CHKERRCONTINUE(ierr);

    // User gives threads to PETSc for threaded PETSc work
    ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
    if(prank>=0) {

      // Vec work
      ierr = VecScale(y,2.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);

    // User gives threads to PETSc for threaded PETSc work
    ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
    if(prank>=0) {

      PetscScalar *avals, *bvals;
      ierr = PetscMalloc1(n,&avals);CHKERRCONTINUE(ierr);
      ierr = PetscMalloc1(n,&bvals);CHKERRCONTINUE(ierr);
      PetscInt pstart,pend;
      ierr = VecGetOwnershipRange(x,&pstart,&pend);CHKERRCONTINUE(ierr);
      for(i=0; i<lsize; i++) {
        avals[i] = pstart + i + 1.0;
        bvals[i] = -n+5 + (pstart + i)*2.0;
      }
      ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,PETSC_DECIDE,lsize,n,avals,&a);CHKERRCONTINUE(ierr);
      ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,PETSC_DECIDE,lsize,n,bvals,&b);CHKERRCONTINUE(ierr);
      //VecView(b,PETSC_VIEWER_STDOUT_WORLD);

      // Vec reductions
      PetscInt vminind, vmaxind;
      PetscScalar vmin, vmax;
      ierr = VecMin(a,&vminind,&vmin);CHKERRCONTINUE(ierr);
      ierr = VecMax(a,&vmaxind,&vmax);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Min=%f Max=%f\n",vmin,vmax);CHKERRCONTINUE(ierr);

      // Vec Pointwise
      ierr = VecPointwiseMult(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"PMult Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecPointwiseDivide(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"PDiv Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecPointwiseMax(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"PMax Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecPointwiseMin(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"PMin Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecPointwiseMaxAbs(c,a,b);CHKERRCONTINUE(ierr);
      ierr = VecNorm(c,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"PMaxAbs Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      // Vec multiple dot product
      VecSet(c,5.0);
      Vec mvecs[2];
      PetscScalar vals[2];
      mvecs[0] = a;
      mvecs[1] = b;
      VecMDot(c,2,mvecs,vals);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"mdot n1=%f n2=%f\n",vals[0],vals[1]);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);
  }

  // Destroy Vecs
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vec destroy\n");CHKERRCONTINUE(ierr);
  ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
  ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
  ierr = VecDestroy(&a);CHKERRCONTINUE(ierr);
  ierr = VecDestroy(&b);CHKERRCONTINUE(ierr);

  PetscFinalize();
  return 0;
}
