static char help[] = "Test threadcomm with PThreads thread type and user threading model with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

void func(void *arg);

Vec x, y;
PetscScalar *ay;
MPI_Comm comm;

/*
   Example run command: ./ex11 -n 1000 -threadcomm_type pthread -threadcomm_model user -threadcomm_nthreads $nthreads
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt nthreads, tnum;
  pthread_t *tid;
  pthread_attr_t *attr;
  PetscInt *tranks;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_NULL,&comm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm,&nthreads);CHKERRQ(ierr);

  ierr = PetscMalloc1(nthreads,&tid);CHKERRQ(ierr);
  ierr = PetscMalloc1(nthreads,&attr);CHKERRQ(ierr);
  ierr = PetscMalloc1(nthreads,&tranks);CHKERRQ(ierr);

  /* Create pthreads */
  ierr = PetscPrintf(comm,"Creating %d threads\n",nthreads);CHKERRQ(ierr);
  for(tnum=0; tnum<nthreads; tnum++) {
    ierr = pthread_attr_init(&attr[tnum]);
    tranks[tnum] = tnum;
    pthread_create(&tid[tnum],&attr[tnum],(void*)func,&tranks[tnum]);
  }

  /* Join pthreads */
  void *res;
  for(tnum=0; tnum<nthreads; tnum++) {
    pthread_attr_destroy(&attr[tnum]);
    pthread_join(tid[tnum],&res);
  }
  ierr = PetscPrintf(comm,"Joined threads\n");

  /* Destroy data structures */
  ierr = PetscFree(tid);CHKERRQ(ierr);
  ierr = PetscFree(attr);CHKERRQ(ierr);
  ierr = PetscFree(tranks);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* User function called by all threads in parallel */
void func(void *arg) {

  PetscInt i, n=100, commrank=0, start, end, lsize, *indices;
  PetscScalar alpha=3.0, vnorm;
  int trank = *(int*)arg;
  PetscErrorCode ierr;

  ierr = PetscThreadInitialize();CHKERRCONTINUE(ierr);

  ierr = PetscPrintf(comm,"Running user function tests\n");CHKERRCONTINUE(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRCONTINUE(ierr);

  /* User gives threads to PETSc to use for PETSc functions */
  ierr = PetscThreadCommJoinComm(comm,trank,&commrank);CHKERRCONTINUE(ierr);
  if(commrank>=0) {
    ierr = VecCreate(comm,&x);CHKERRCONTINUE(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
    ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
    ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

    ierr = VecSet(x,2.0);CHKERRCONTINUE(ierr);
    ierr = VecSet(y,3.0);CHKERRCONTINUE(ierr);
    ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);

    ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
    ierr = PetscPrintf(comm,"Test1 Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);
  }
  /* User takes back threads from PETSc once done calling PETSc functions */
  ierr = PetscThreadCommReturnComm(comm,trank,&commrank);CHKERRCONTINUE(ierr);

  /* Get data for local work */
  ierr = VecGetArray(y,&ay);CHKERRCONTINUE(ierr);
  ierr = VecGetLocalSize(x,&lsize);CHKERRCONTINUE(ierr);
  ierr = PetscThreadCommGetOwnershipRanges(comm,lsize,&indices);CHKERRCONTINUE(ierr);

  /* Parallel threaded user code */
  start = indices[trank];
  end = indices[trank+1];
  for(i=start; i<end; i++) ay[i] = ay[i]*ay[i];

  /* Restore vector */
  ierr = VecRestoreArray(y,&ay);CHKERRCONTINUE(ierr);

  /* User gives threads to PETSc for threaded PETSc work */
  ierr = PetscThreadCommJoinComm(comm,trank,&commrank);CHKERRCONTINUE(ierr);
  if(commrank>=0) {
    /* Vec work */
    ierr = VecScale(y,2.0);CHKERRCONTINUE(ierr);
    ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);

    ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
    ierr = PetscPrintf(comm,"Test2 Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

    ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
    ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
  }
  ierr = PetscThreadCommReturnComm(comm,trank,&commrank);CHKERRCONTINUE(ierr);

  ierr = PetscFree(indices);CHKERRCONTINUE(ierr);
  ierr = PetscThreadFinalize();CHKERRCONTINUE(ierr);
}
