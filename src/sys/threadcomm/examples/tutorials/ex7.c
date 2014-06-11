static char help[] = "Test PetscThreadPool with pthreads with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>
#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

void func(void *arg);

Vec x, y;
PetscInt *indices;
PetscScalar *ay;

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

  printf("Calling PetscFinalize\n");
  PetscFinalize();

  return 0;
}

void func(void *arg) {

  PetscInt i, n=100, prank, start, end, nthreads;
  PetscScalar alpha=3.0, vnorm;
  int trank = *(int*)arg;
  PetscErrorCode ierr;

  printf("in func trank=%d\n",trank);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRCONTINUE(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRCONTINUE(ierr);

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

    printf("Vec set\n");
    ierr = VecSet(x,2.0);CHKERRCONTINUE(ierr);
    ierr = VecSet(y,3.0);CHKERRCONTINUE(ierr);
    ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);

    //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
    printf("Norm=%f\n",vnorm);
    ierr = VecGetArray(y,&ay);CHKERRCONTINUE(ierr);
    PetscThreadCommGetOwnershipRanges(PETSC_COMM_WORLD,n,&indices);
  }
  // User takes back threads from PETSc once done calling PETSc functions
  ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);

  // Insert parallel threaded user code here
  start = indices[trank];
  end = indices[trank+1];
  printf("trank=%d nthreads=%d start=%d end=%d\n",trank,nthreads,start,end);
  for(i=start; i<end; i++) {
    ay[i] = ay[i]*ay[i];
  }

  printf("\n\n\nReturning threads to petsc\n");
  ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
  if(prank>=0) {

    ierr = VecRestoreArray(y,&ay);CHKERRCONTINUE(ierr);
    VecScale(y,2.0);
    VecAXPY(y,alpha,x);

    //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
    printf("Norm=%f\n",vnorm);

    printf("Vec destroy\n");
    ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
    ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
  }
   ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);
}
