#include <petsc-private/threadcommimpl.h>

/* Global variables for thread model/thread type */
PetscInt ThreadModel;
PetscInt ThreadType;

/* Thread specific global variables */
#if defined(PETSC_HAVE_PTHREADCLASSES)
#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscInt PetscMasterThread;
PETSC_PTHREAD_LOCAL PetscInt PetscThreadInit;
#else
PetscThreadKey PetscMasterThread;
PetscThreadKey PetscThreadInit;
#endif
#elif defined(PETSC_HAVE_OPENMP)
PetscInt PetscMasterThread;
PetscInt PetscThreadInit;
#pragma omp threadprivate(PetscMasterThread,PetscThreadInit)
#else
PetscInt PetscMasterThread;
PetscInt PetscThreadInit;
#endif

/* Lock functions */
PetscErrorCode (*PetscThreadLockAcquire)(void*);
PetscErrorCode (*PetscThreadLockRelease)(void*);
PetscErrorCode (*PetscThreadLockCreate)(void**);
PetscErrorCode (*PetscThreadLockDestroy)(void**);

/* Extern variables/functions */
PETSC_EXTERN PetscBool PetscThreadCommRegisterAllModelsCalled;
PETSC_EXTERN PetscBool PetscThreadCommRegisterAllTypesCalled;
PETSC_EXTERN PetscErrorCode PetscThreadCommStackCreate();
PETSC_EXTERN PetscErrorCode PetscThreadCommStackDestroy();
#if defined(PETSC_USE_LOG)
PETSC_EXTERN PetscObject *PetscObjects;
PETSC_EXTERN PetscInt    PetscObjectsCounts, PetscObjectsMaxCounts;
#endif
PETSC_EXTERN PetscErrorCode PetscSetUseTrMalloc_Private(void);

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize"
/*@
   PetscThreadInitialize - Initialize thread specific data structs

   Not Collective

   Level: Advanced

   Notes:
   Initialize thread specific global variables and data structures.
   Must be called after thread is created to take advantage of PETSc
   system routines.

.seealso PetscThreadFinalize()
@*/
PetscErrorCode PetscThreadInitialize(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Make sure this thread has not been initialized yet */
  if(PetscMasterThread || PetscThreadInit) PetscFunctionReturn(0);

  /* Create thread stack */
  ierr = PetscThreadCommStackCreate();CHKERRQ(ierr);
  /* Setup TRMalloc */
  ierr = PetscTrMallocInitialize();CHKERRQ(ierr);
  PetscThreadInit = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize"
/*@
   PetscThreadFinalize - Merge thread specific data with the main thread and
                         destroy thread specific data structs

   Not Collective

   Level: Advanced

   Notes:
   Finalize thread specific global variables and data structures. If needed
   merge data from this thread with main thread/global variables. Must be
   called before thread is destroyed.

.seealso PetscThreadInitialize()
@*/
PetscErrorCode PetscThreadFinalize(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(PetscMasterThread || !PetscThreadInit) PetscFunctionReturn(0);

  /* Add code to destroy TRMalloc/merged with main trmalloc data */
  ierr = PetscTrMallocFinalize();CHKERRQ(ierr);
  /* Destroy thread stack */
  ierr = PetscThreadCommStackDestroy();CHKERRQ(ierr);
  PetscThreadInit = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadSetModel"
/*@C
   PetscThreadPoolSetModel - Sets the threading model for the thread communicator

   Logically collective

   Input Parameters:
+  tcomm - the thread communicator
-  model  - the type of thread model needed

   Options Database keys:
   -threadcomm_model <type>

   Available models
   See "petsc/include/petscthreadcomm.h" for available types

   Level: developer

   Notes:
   Sets threading model for the threadpool by checking thread model function list and calling
   the appropriate function.

@*/
PetscErrorCode PetscThreadSetModel(PetscThreadCommModel model)
{
  PetscErrorCode ierr,(*r)();
  char           smodel[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidCharPointer(model,2);
  if (!PetscThreadCommRegisterAllModelsCalled) { ierr = PetscThreadCommRegisterAllModels();CHKERRQ(ierr);}

  /* Get thread model from command line */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Threadcomm model - setting threading model",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-threadcomm_model","Threadcomm model","PetscThreadCommSetModel",PetscThreadCommModelList,model,smodel,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!flg) ierr = PetscStrcpy(smodel,model);CHKERRQ(ierr);

  /* Find and call thread model initialization function */
  ierr = PetscFunctionListFind(PetscThreadCommModelList,smodel,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested thread model %s",smodel);
  ierr = (*r)();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadSetType"
/*@C
   PetscThreadPoolSetType - Sets the threading type for the thread communicator

   Logically Collective

   Input Parameters:
+  tcomm - the thread communicator
-  type  - the type of thread model needed

   Options Database keys:
   -threadcomm_type <type>

   Available types
   See "petsc/include/petscthreadcomm.h" for available types

   Level: developer

   Notes:
   Sets type of threadpool by checking thread type function list and calling
   the appropriate function.

@*/
PetscErrorCode PetscThreadSetType(PetscThreadCommType type)
{
  PetscBool      flg;
  PetscErrorCode ierr,(*r)();
  char           stype[256];

  PetscFunctionBegin;
  PetscValidCharPointer(type,2);
  if (!PetscThreadCommRegisterAllTypesCalled) { ierr = PetscThreadCommRegisterAllTypes();CHKERRQ(ierr);}

  /* Get thread type from command line */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Threadcomm type - setting threading type",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-threadcomm_type","Threadcomm type","PetscThreadCommSetType",PetscThreadTypeList,type,stype,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!flg) ierr = PetscStrcpy(stype,type);CHKERRQ(ierr);

  /* Find and call thread type initialization function */
  ierr = PetscFunctionListFind(PetscThreadTypeList,stype,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested thread type %s",stype);
  ierr = (*r)();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLocksInitialize"
/*@
   PetscThreadLocksInitialize - Allocate and initialize all locks

   Logically Collective

   Level: developer

   Notes:
   Calls implementation specific routine to allocate and initialize
   each lock.

@*/
PetscErrorCode PetscThreadLocksInitialize(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscLocks) PetscFunctionReturn(0);
  ierr = PetscNew(&PetscLocks);CHKERRQ(ierr);
  ierr = (*PetscThreadLockCreate)(&PetscLocks->trmalloc_lock);CHKERRQ(ierr);
  ierr = (*PetscThreadLockCreate)(&PetscLocks->vec_lock);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLocksFinalize"
/*@
   PetscThreadLocksFinalize - Destroy and free all locks

   Logically Collective

   Level: developer

   Notes:
   Calls implementation specific routine to destroy each lock.

@*/
PetscErrorCode PetscThreadLocksFinalize(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!PetscLocks) PetscFunctionReturn(0);
  ierr = (*PetscThreadLockDestroy)(PetscLocks->trmalloc_lock);CHKERRQ(ierr);
  ierr = (*PetscThreadLockDestroy)(PetscLocks->vec_lock);CHKERRQ(ierr);
  ierr = PetscFree(PetscLocks);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
