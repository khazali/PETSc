#include <petsc-private/threadcommimpl.h>

/* Variables to track package registration/initialization */
static PetscBool       PetscThreadCommPackageInitialized = PETSC_FALSE;
PETSC_EXTERN PetscBool PetscThreadCommRegisterAllModelsCalled;
PETSC_EXTERN PetscBool PetscThreadCommRegisterAllTypesCalled;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommFinalizePackage"
/*@C
   PetscThreadCommFinalizePackage - Finalize PetscThreadComm package, called from PetscFinalize()

   Logically collective

   Level: developer

.seealso: PetscThreadCommInitializePackage()
@*/
PetscErrorCode PetscThreadCommFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy function lists and thread key value */
  ierr = PetscFunctionListDestroy(&PetscThreadTypeList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscThreadPoolTypeList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscThreadCommTypeList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscThreadCommModelList);CHKERRQ(ierr);
  ierr = MPI_Keyval_free(&Petsc_ThreadComm_keyval);CHKERRQ(ierr);

  /* Set global variables to specific that threadcomm package is not set up */
  PetscThreadCommPackageInitialized      = PETSC_FALSE;
  PetscThreadCommRegisterAllModelsCalled = PETSC_FALSE;
  PetscThreadCommRegisterAllTypesCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_CopyThreadComm"
/*@C
   Petsc_CopyThreadComm - Copies the thread communicator attached to MPI_Comm

   Not Collective

   Input Parameters:
+  comm        - MPI communicator
.  keyval      - Not used
.  extra_state - Not used
.  attr_in     - Threadcomm to copy
-  attr_out    - Copy of threadcomm

   Output Parameters:
.  flag        - True if threadcomm is copied

   Level: developer

   Notes:
   This is called by MPI, not by users. This is called when MPI_Comm_dup() is called on the
   communicator. This is declared extern "C" because it is passed to MPI_Keyval_create()

.seealso Petsc_DelThreadComm()
*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_CopyThreadComm(MPI_Comm comm,PetscMPIInt keyval,void *extra_state,void *attr_in,void *attr_out,int *flag)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm;

  PetscFunctionBegin;
  tcomm = (PetscThreadComm)attr_in;
  tcomm->refct++;
  *(void**)attr_out = tcomm;
  *flag = 1;
  ierr  = PetscInfo1(0,"Copying thread communicator data in an MPI_Comm %ld\n",(long)comm);CHKERRQ(ierr);
  if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_DelThreadComm"
/*@C
   Petsc_DelThreadComm - Frees the thread communicator attached to MPI_Comm

   Not Collective

   Input Parameters:
+  comm        - MPI communicator
.  keyval      - Not used
.  attr        - Threadcomm to free
-  extra_state - Not used

   Level: developer

   Notes:
   This is called by MPI, not by users. This is called when MPI_Comm_free() is called on
   the communicator. This is declared extern "C" because it is passed to MPI_Keyval_create()

.seealso Petsc_CopyThreadComm()
@*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelThreadComm(MPI_Comm comm,PetscMPIInt keyval,void *attr,void *extra_state)
{
  PetscThreadComm tcomm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  tcomm = (PetscThreadComm)attr;
  ierr = PetscThreadCommDestroy((PetscThreadComm*)&attr);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Deleting thread communicator data in an MPI_Comm %ld\n",(long)comm);
  if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitializePackage"
/*@C
   PetscThreadCommInitializePackage - Initializes ThreadComm package

   Logically collective

   Level: developer

   Notes:
   Create the needed keyvalues and register events and packages for threadcomm routines. 

.seealso: PetscThreadCommFinalizePackage()
@*/
PetscErrorCode PetscThreadCommInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscThreadCommPackageInitialized) PetscFunctionReturn(0);

  if (Petsc_ThreadComm_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(Petsc_CopyThreadComm,Petsc_DelThreadComm,&Petsc_ThreadComm_keyval,(void*)0);CHKERRQ(ierr);
  }

  ierr = PetscGetNCores(NULL);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ThreadCommRunKer",  0, &ThreadComm_RunKernel);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ThreadCommBarrie",  0, &ThreadComm_Barrier);CHKERRQ(ierr);

  PetscThreadCommPackageInitialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PetscThreadCommFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
