static char help[] = "Test threadcomm with type=pthreads,model=user with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>
#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

void func(void *arg);

Vec x, y;
PetscScalar *ay;
MPI_Comm comm;

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

  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,PETSC_DECIDE,&comm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm,&nthreads);CHKERRQ(ierr);

  tid = (pthread_t*)malloc(sizeof(pthread_t)*nthreads);
  attr = (pthread_attr_t*)malloc(sizeof(pthread_attr_t)*nthreads);
  tranks = (PetscInt*)malloc(sizeof(PetscInt)*nthreads);

  printf("Creating %d threads\n",nthreads);
  for(tnum=0; tnum<nthreads; tnum++) {
    printf("Creating thread %d\n",tnum);
    ierr = pthread_attr_init(&attr[tnum]);
    tranks[tnum] = tnum;
    pthread_create(&tid[tnum],&attr[tnum],(void*)func,&tranks[tnum]);
  }

  void *res;
  for(tnum=0; tnum<nthreads; tnum++) {
    printf("Joining thread %d\n",tnum);
    pthread_join(tid[tnum],&res);
  }

  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}

void func(void *arg) {

  PetscInt i, n=100, commrank, start, end, lsize, *indices;
  PetscScalar alpha=3.0, vnorm;
  int trank = *(int*)arg;
  PetscErrorCode ierr;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"in func trank=%d\n",trank);CHKERRCONTINUE(ierr);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRCONTINUE(ierr);

  // User gives threads to PETSc to use for PETSc functions
  ierr = PetscThreadCommJoin(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);
  ierr = PetscPrintf(comm,"rank=%d joined comm commrank=%d\n",trank,commrank);CHKERRCONTINUE(ierr);
  if(commrank>=0) {
    ierr = VecCreate(comm,&x);CHKERRCONTINUE(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
    ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
    ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

    ierr = PetscPrintf(comm,"Vec work\n");CHKERRCONTINUE(ierr);
    ierr = VecSet(x,2.0);CHKERRCONTINUE(ierr);
    ierr = VecSet(y,3.0);CHKERRCONTINUE(ierr);
    ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);

    //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
    ierr = PetscPrintf(comm,"Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);
  }
  // User takes back threads from PETSc once done calling PETSc functions
  ierr = PetscThreadCommReturn(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);

  // Get data for local work
  ierr = VecGetArray(y,&ay);CHKERRCONTINUE(ierr);
  ierr = VecGetLocalSize(x,&lsize);CHKERRCONTINUE(ierr);
  ierr = PetscThreadCommGetOwnershipRanges(comm,lsize,&indices);CHKERRCONTINUE(ierr);

  // Parallel threaded user code
  start = indices[trank];
  end = indices[trank+1];
  for(i=start; i<end; i++) {
    ay[i] = ay[i]*ay[i];
  }

  // Restore vector
  ierr = VecRestoreArray(y,&ay);CHKERRCONTINUE(ierr);

  // User gives threads to PETSc for threaded PETSc work
  ierr = PetscThreadCommJoin(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);

  if(commrank>=0) {
    // Vec work
    ierr = VecScale(y,2.0);CHKERRCONTINUE(ierr);
    ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);

    //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
    ierr = PetscPrintf(comm,"Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

    ierr = PetscPrintf(comm,"Vec destroy\n");
    ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
    ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
  }
  ierr = PetscThreadCommReturn(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);
}
