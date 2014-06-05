static char help[] = "Test PetscThreadPool with pthreads.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec             x;
  PetscErrorCode  ierr;
  PetscInt        tnum, prank, nthreads, n=20;
  PetscScalar     one = 1.0;
  PetscThreadPool pool;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  printf("nthreads=%d\n",nthreads);

  PetscThreadPoolExecutePthread(PETSC_COMM_WORLD,&parmainfunc);

  PetscFinalize();
}

// This would be hidden from user in PETSc
// Note that PETSc MPI_Comm contains threadcomm and threadcomm contains threadpools
int PetscThreadPoolExecutePthread(void *func) {

  pthread_t *tid = (pthread_t*)malloc(sizeof(pthread_t)*nthreads);
  pthread_attr_t attr;
  pthread_t_attr_init(&attr);
  for(tnum=0; tnum<nthreads; tnum++) {
    pthread_create(tid[tnum],&attr,PetscThreadCommInit_PThread,&tnum);
  }

  func(0);

  void *res;
  for(tnum=0; tnum<nthreads; tnum++) {
    pthread_join(tid[tnum],&res);
  }

  return 0;
}

void* parmainfunc(void *arg) {
  int trank = (int)arg;
  int npools = 1;

  // Insert parallel threaded user code here

  // User gives threads to PETSc to use for PETSc functions
  ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);
  printf("rank=%d joined pool prank=%d\n",trank,prank);
  if(prank) {
    printf("rank=%d working on vec\n",trank);
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRCONTINUE(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
    ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);

    printf("vec set\n");
    ierr = VecSet(x,one);CHKERRCONTINUE(ierr);

    //PetscThreadPoolBarrier(pool);
    VecView(x,PETSC_VIEWER_STDOUT_WORLD);

    printf("vec destroy\n");
    ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
  }

  // User takes back threads from PETSc once done calling PETSc functions
  ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);

  // Insert parallel threaded user code here

}
