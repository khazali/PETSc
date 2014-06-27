#include <../src/sys/threadcomm/impls/nothread/nothreadimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_NoThread"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm comm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (comm->ncommthreads != 1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Cannot have more than 1 thread for the nonthread communicator,threads requested = %D",comm->ncommthreads);
  ierr = PetscStrcpy(comm->type,NOTHREAD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
