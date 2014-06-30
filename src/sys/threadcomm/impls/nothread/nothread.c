#include <../src/sys/threadcomm/impls/nothread/nothreadimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInit_NoThread"
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_NoThread(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(pool->type,NOTHREAD);CHKERRQ(ierr);
  pool->threadtype = THREAD_TYPE_NOTHREAD;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_NoThread"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm comm)
{
  PetscFunctionBegin;
  if (comm->ncommthreads != 1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Cannot have more than 1 thread for the nonthread communicator,threads requested = %D",comm->ncommthreads);
  PetscFunctionReturn(0);
}
