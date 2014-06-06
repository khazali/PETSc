static char help[] = "Test PetscThreadPool with pthreads.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>
#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

void func(void *arg);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt        n=100;
  PetscInt nthreads=8, tnum;
  pthread_t *tid;
  pthread_attr_t *attr;
  PetscInt *tranks;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);

  tid = (pthread_t*)malloc(sizeof(pthread_t)*nthreads);
  attr = (pthread_attr_t*)malloc(sizeof(pthread_attr_t)*nthreads);
  tranks = (PetscInt*)malloc(sizeof(PetscInt)*nthreads);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  for(tnum=0; tnum<nthreads; tnum++) {
    printf("Creating thread %d\n",tnum);
    ierr = pthread_attr_init(&attr[tnum]);
    tranks[tnum] = tnum;
    pthread_create(&tid[tnum],&attr[tnum],(void*)func,&tranks[tnum]);
  }

  void *res;
  for(tnum=0; tnum<nthreads; tnum++) {
    pthread_join(tid[tnum],&res);
  }

  PetscFinalize();

  return 0;
}

void func(void *arg) {

  Vec x;
  PetscInt n=100, prank;
  int trank = *(int*)arg;
  PetscErrorCode ierr;

  printf("in func trank=%d\n",trank);

  // Insert parallel threaded user code here

  // User gives threads to PETSc to use for PETSc functions
  ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
  printf("rank=%d joined pool prank=%d\n",trank,prank);
  if(prank) {
    printf("rank=%d working on vec\n",trank);
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRCONTINUE(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
    ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);

    printf("vec set\n");
    ierr = VecSet(x,1.0);CHKERRCONTINUE(ierr);

    //PetscThreadPoolBarrier(pool);
    VecView(x,PETSC_VIEWER_STDOUT_WORLD);

    printf("vec destroy\n");
    ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
  }

  // User takes back threads from PETSc once done calling PETSc functions
  ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);

  // Insert parallel threaded user code here
}
