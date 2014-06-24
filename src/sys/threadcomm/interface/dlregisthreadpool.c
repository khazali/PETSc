#include <petsc-private/threadcommimpl.h>

static PetscBool PetscThreadPoolPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolFinalizePackage"
/*@C
   PetscThreadPoolFinalizePackage - Finalize PetscThreadPool package, called from PetscFinalize()

   Logically collective

   Level: developer

.seealso: PetscThreadPoolInitializePackage()
@*/
PetscErrorCode PetscThreadPoolFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscThreadPoolPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_CopyThreadPool"
/*
  This frees the thread communicator attached to MPI_Comm

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Keyval_create()
*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_CopyThreadPool(MPI_Comm comm,PetscMPIInt keyval,void *extra_state,void *attr_in,void *attr_out,int *flag)
{
  PetscErrorCode  ierr;
  PetscThreadPool pool = (PetscThreadPool)attr_in;

  PetscFunctionBegin;
  pool->refct++;
  *(void**)attr_out = pool;

  *flag = 1;
  ierr  = PetscInfo1(0,"Copying thread pool data in an MPI_Comm %ld\n",(long)comm);CHKERRQ(ierr);
  if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_DelThreadPool"
/*
  This frees the thread communicator attached to MPI_Comm

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Keyval_create()
*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelThreadPool(MPI_Comm comm,PetscMPIInt keyval,void *attr,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscThreadPoolDestroy((PetscThreadComm)&attr);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Deleting thread pool data in an MPI_Comm %ld\n",(long)comm);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInitializePackage"
/*@C
   PetscThreadPoolInitializePackage - Initializes threadpool package

   Logically collective

   Level: developer

.seealso: PetscThreadPoolFinalizePackage()
@*/
PetscErrorCode PetscThreadPoolInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscThreadPoolPackageInitialized) PetscFunctionReturn(0);

  ierr = PetscGetNCores(NULL);CHKERRQ(ierr);

  PetscThreadPoolPackageInitialized = PETSC_TRUE;

  ierr = PetscRegisterFinalize(PetscThreadPoolFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
