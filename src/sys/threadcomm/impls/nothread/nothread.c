#include <../src/sys/threadcomm/impls/nothread/nothreadimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_NoThread"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm tcomm)
{
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tcomm->ncommthreads != 1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Cannot have more than 1 thread for the nonthread communicator,threads requested = %D",tcomm->ncommthreads);
  ierr = PetscStrcpy(pool->type,NOTHREAD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
