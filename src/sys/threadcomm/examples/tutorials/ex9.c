static char help[] = "Test PetscThreadPool with pthreads.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>
#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

PetscErrorCode PetscThreadCommExecute_Pthread(void *func);
void parmainfunc(void *arg);
typedef void (*func_type)(void *arg);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt        n=20;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  PetscThreadCommExecute_Pthread(parmainfunc);
  PetscFinalize();

  return 0;
}

// This would be hidden from user in PETSc
// Note that PETSc MPI_Comm contains threadcomm and threadcomm contains threadpools
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommExecute_Pthread"
PetscErrorCode PetscThreadCommExecute_Pthread(void *func) {

  PetscErrorCode ierr;
  PetscInt nthreads=8, tnum;
  pthread_t *tid;
  pthread_attr_t *attr;
  PetscInt *tranks;

  PetscFunctionBegin;
  tid = (pthread_t*)malloc(sizeof(pthread_t)*nthreads);
  attr = (pthread_attr_t*)malloc(sizeof(pthread_attr_t)*nthreads);
  tranks = (PetscInt*)malloc(sizeof(PetscInt)*nthreads);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  for(tnum=1; tnum<nthreads; tnum++) {
    printf("Creating thread %d\n",tnum);
    ierr = pthread_attr_init(&attr[tnum]);
    tranks[tnum] = tnum;
    pthread_create(&tid[tnum],&attr[tnum],func,&tranks[tnum]);
  }

  PetscInt rank=0;
  //((void(*)(void*))func)((void*)&rank);
  ((func_type)func)((void*)&rank);

  void *res;
  for(tnum=0; tnum<nthreads; tnum++) {
    pthread_join(tid[tnum],&res);
  }
  PetscFunctionReturn(0);
}

void parmainfunc(void *arg) {

  Vec x;
  PetscInt n=100, prank;
  int trank = *(int*)arg;
  PetscErrorCode ierr;

  printf("in parmainfunc trank=%d\n",trank);

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
