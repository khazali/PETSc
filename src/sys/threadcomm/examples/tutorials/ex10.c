static char help[] = "Test PetscThreadPool with pthreads.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>
#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

void func(void *arg);

pthread_barrier_t barr;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt nthreads, tnum;
  pthread_t *tid;
  pthread_attr_t *attr;
  PetscInt *tranks;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);

  tid = (pthread_t*)malloc(sizeof(pthread_t)*nthreads);
  attr = (pthread_attr_t*)malloc(sizeof(pthread_attr_t)*nthreads);
  tranks = (PetscInt*)malloc(sizeof(PetscInt)*nthreads);

  pthread_barrier_init(&barr,NULL,nthreads);

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
  pthread_barrier_destroy(&barr);

  printf("Calling PetscFinalize\n");
  PetscFinalize();

  return 0;
}

void func(void *arg) {

  Vec x, y;
  PetscScalar *ay;
  PetscInt i, n=100, prank, start, end, nthreads=4;
  PetscScalar alpha, vnorm;
  int trank = *(int*)arg;
  PetscErrorCode ierr;

  printf("in func trank=%d\n",trank);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRCONTINUE(ierr);

  // Insert parallel threaded user code here

  // User gives threads to PETSc to use for PETSc functions
  ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
  printf("rank=%d joined pool prank=%d\n",trank,prank);
  if(prank>=0) {
    printf("rank=%d working on vec\n",trank);
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRCONTINUE(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
    ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
    ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

    printf("vec set\n");
    alpha = 3.0;
    ierr = VecSet(x,1.0);CHKERRCONTINUE(ierr);
    ierr = VecSet(y,2.0);CHKERRCONTINUE(ierr);
    ierr = VecAXPY(y,alpha,x);

    // Barrier needed if not synchronizing after each call to print correctly
    //PetscThreadPoolBarrier(PETSC_THREAD_COMM_WORLD);
    //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
    VecNorm(y,NORM_2,&vnorm);
    printf("Norm=%f\n",vnorm);
  }

  // User takes back threads from PETSc once done calling PETSc functions
  ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);

  printf("\n\n\nUser modifying code\n");
  // Insert parallel threaded user code here
  VecGetArray(y,&ay);
  start = trank*(n/nthreads);
  end = trank*(n/nthreads);
  for(i=start; i<end; i++) {
    ay[i] = ay[i]*ay[i];
  }
  VecRestoreArray(y,&ay);
  pthread_barrier_wait(&barr);

  printf("\n\n\nReturning threads to petsc\n");
  ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
  if(prank>=0) {
    VecNorm(y,NORM_2,&vnorm);
    printf("Norm=%f\n",vnorm);

    printf("vec destroy\n");
    ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
    ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
  }
  ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);
}
