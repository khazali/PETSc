#include <../src/sys/threadcomm/impls/nothread/nothreadimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInit_NoThread"
PETSC_EXTERN PetscErrorCode PetscThreadInit_NoThread()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ThreadType = THREAD_TYPE_NOTHREAD;
  ierr = PetscNew(&PetscLocks);CHKERRQ(ierr);
  PetscLocks->trmalloc_lock = PETSC_NULL;
  PetscThreadLockAcquire = PetscThreadLockAcquire_NoThread;
  PetscThreadLockRelease = PetscThreadLockRelease_NoThread;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInit_NoThread"
/*
   PetscThreadPoolInit_NoThread - Initialize a threadpool to use the
                                  nothread thread type

   Not Collective

   Input Parameters:
.  pool - Threadpool to initialize

   Level: developer

*/
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_NoThread(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(pool->type,NOTHREAD);CHKERRQ(ierr);
  pool->threadtype = THREAD_TYPE_NOTHREAD;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInit_NoThread"
/*
   PetscThreadCommInit_NoThread - Initialize a threadcomm to use the
                                  nothread thread type

   Not Collective

   Input Parameters:
.  comm - Threadcomm to initialize

   Level: developer

*/
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_NoThread(PetscThreadComm tcomm)
{
  PetscFunctionBegin;
  if (tcomm->ncommthreads != 1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Cannot have more than 1 thread for the nonthread communicator,threads requested = %D",tcomm->ncommthreads);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockAcquire_NoThread"
PetscErrorCode PetscThreadLockAcquire_NoThread(void *lock)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockRelease_NoThread"
PetscErrorCode PetscThreadLockRelease_NoThread(void *lock)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
