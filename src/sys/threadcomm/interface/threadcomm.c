/* Define feature test macros to make sure CPU_SET and other functions are available */
#define PETSC_DESIRE_FEATURE_TEST_MACROS

#include <petsc-private/threadcommimpl.h>      /*I "petscthreadcomm.h" I*/
#include <petscviewer.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

PetscMPIInt       Petsc_ThreadComm_keyval     = MPI_KEYVAL_INVALID;
PetscFunctionList PetscThreadCommInitTypeList = PETSC_NULL;
PetscFunctionList PetscThreadCommTypeList     = PETSC_NULL;
PetscFunctionList PetscThreadCommModelList    = PETSC_NULL;

extern PetscInt   N_CORES;

/* Logging support */
PetscLogEvent ThreadComm_RunKernel, ThreadComm_Barrier;

static PetscErrorCode PetscThreadCommRunKernel0_Private(PetscThreadComm tcomm,PetscErrorCode (*func)(PetscInt,...));

#undef __FUNCT__
#define __FUNCT__ "PetscCommGetThreadComm"
/*@C
   PetscCommGetThreadComm - Gets the thread communicator
                            associated with the MPI communicator

   Not Collective

   Input Parameters:
.  comm - MPI communicator

   Output Parameters:
.  tcommp - Pointer to the thread communicator

   Level: Intermediate

   Notes: If no thread communicator is on the MPI_Comm then the global
   thread communicator is returned.

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy()
@*/
PetscErrorCode PetscCommGetThreadComm(MPI_Comm comm,PetscThreadComm *tcomm)
{
  PetscErrorCode  ierr;
  PetscMPIInt     flg;
  void            *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,(PetscThreadComm*)&ptr,&flg);CHKERRQ(ierr);
  if (!flg) {
    // Create and attach threadcomm based on user input options
    PetscThreadCommCreateAttach(comm,PETSC_DECIDE,PETSC_NULL,PETSC_NULL);
    // Get threadcomm from MPI_Comm
    ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,(PetscThreadComm*)&ptr,&flg);CHKERRQ(ierr);
  }
  // Return threadcomm or return error message
  if(flg) *tcomm = (PetscThreadComm)ptr;
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to attach threadcomm to MPI_Comm.");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCommCheckGetThreadComm"
/*
   PetscCommCheckGetThreadComm - Check if threadcomm exists and if so return it.

   Not Collective

   Input Parameters:
.  comm - MPI Communicator

   Output Parameters:
+  tcomm - Thread communicator attached to the MPI_Comm
-  set   - True if threadcomm exists, false if not

   Level: developer

   Notes:
   Tries to get the threadcomm from the MPI comm. If successful returns the
   threadcomm and a flag set to true, if not returns PETSC_NULL and a flag
   set to false.

*/
PetscErrorCode PetscCommCheckGetThreadComm(MPI_Comm comm,PetscThreadComm *tcomm,PetscBool *set)
{
  PetscMPIInt    flg;
  PetscErrorCode ierr;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,(PetscThreadComm*)&ptr,&flg);CHKERRQ(ierr);
  if (flg) {
    *set   = PETSC_TRUE;
    *tcomm = (PetscThreadComm)ptr;
  } else {
    *set   = PETSC_FALSE;
    *tcomm = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommAlloc"
/*
   PetscThreadCommAlloc - Allocates a thread communicator object

   Not Collective

   Output Parameters:
.  tcomm - pointer to the thread communicator object

   Level: developer

   Notes:
   Allocates the thread communicator and initializes each variable to
   indicate that the threadcomm has been allocated but still needs to
   be initialized.

.seealso: PetscThreadCommDestroy()
*/
PetscErrorCode PetscThreadCommAlloc(PetscThreadComm *tcomm)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcommout;

  PetscFunctionBegin;
  PetscValidPointer(tcomm,2);

  *tcomm                 = PETSC_NULL;
  ierr                   = PetscNew(&tcommout);CHKERRQ(ierr);

  tcommout->model        = 0;
  tcommout->threadtype   = THREAD_TYPE_NOTHREAD;
  tcommout->nkernels     = 16;
  tcommout->refct        = 0;
  tcommout->lleader      = 0;
  tcommout->gleader      = 0;
  tcommout->thread_start = 0;
  tcommout->red          = PETSC_NULL;
  tcommout->active       = PETSC_FALSE;
  ierr                   = PetscNew(&tcommout->ops);CHKERRQ(ierr);
  tcommout->data         = PETSC_NULL;
  tcommout->syncafter    = PETSC_TRUE;
  tcommout->ismainworker = PETSC_TRUE;

  tcommout->pool         = PETSC_NULL;
  tcommout->ncommthreads = -1;
  tcommout->commthreads  = PETSC_NULL;

  *tcomm                 = tcommout;

  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackCreate"
/*
   PetscThreadCommStackCreate - Create a petscstack on a thread

   Not Collective

   Input Parameters:
.  trank - Thread rank

   Level: developer

   Notes:
   Only used in debug mode. Creates a petscstack on the calling thread.
   Master thread should already have a petscstack on it, so this skips
   the master thread.

*/
PetscErrorCode PetscThreadCommStackCreate(PetscInt trank)
{
  if (trank && !PetscStackActive()) {
    PetscStack *petscstack_in;
    petscstack_in = (PetscStack*)malloc(sizeof(PetscStack));
    petscstack_in->currentsize = 0;
    PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,petscstack_in);
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackDestroy"
/*
   PetscThreadCommStackDestroy - Destroy a petscstack on a thread

   Not Collective

   Input Parameters:
.  trank - Thread rank

   Level: developer

   Notes:
   Only used in debug mode. Destroys the petscstack on the calling thread.
   Master thread will continue being used so do not destroy the petscstack
   on it.

*/
PetscErrorCode PetscThreadCommStackDestroy(PetscInt trank)
{
  if (trank && PetscStackActive()) {
    PetscStack *petscstack_in;
    petscstack_in = (PetscStack*)PetscThreadLocalGetValue(petscstack);
    free(petscstack_in);
    PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,(PetscStack*)0);
  }
  return 0;
}

#else
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackCreate"
/*
   PetscThreadCommStackCreate - Empty function when not using debug mode
*/
PetscErrorCode  PetscThreadCommStackCreate(PetscInt trank)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackDestroy"
/*
   PetscThreadCommStackDestroy - Empty function when not using debug mode
*/
PetscErrorCode  PetscThreadCommStackDestroy(PetscInt trank)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy"
/*
   PetscThreadCommDestroy - Frees a thread communicator object

   Not Collective

   Input Parameters:
.  tcomm - the PetscThreadComm object

   Level: developer

   Notes:
   Reduces the reference count for this threadcomm. Once there are no
   more references to this, the threadcomm is destroyed along with the
   associated threadpool.

.seealso: PetscThreadCommCreate()
*/
PetscErrorCode PetscThreadCommDestroy(PetscThreadComm *tcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*tcomm) PetscFunctionReturn(0);
  if (!--(*tcomm)->refct) {
    printf("Destroying threadcomm\n");
    /* Make sure all jobs are completed */
    ierr = PetscThreadCommJobBarrier(*tcomm);CHKERRQ(ierr);

    /* Destroy pthread specific data */
    if((*tcomm)->threadtype==THREAD_TYPE_PTHREAD) {
      ierr = ((*tcomm)->ops->commdestroy)(*tcomm);CHKERRQ(ierr);
    }

    ierr = PetscThreadPoolDestroy((*tcomm)->pool);CHKERRQ(ierr);
    ierr = PetscThreadCommReductionDestroy((*tcomm)->red);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm)->ops);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm));CHKERRQ(ierr);
  }
  *tcomm = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommView"
/*@C
   PetscThreadCommView - view a thread communicator

   Collective on comm

   Input Parameters:
+  comm - MPI communicator
-  viewer - viewer to display, for example PETSC_VIEWER_STDOUT_WORLD

   Level: developer

.seealso: PetscThreadCommCreate()
@*/
PetscErrorCode PetscThreadCommView(MPI_Comm comm,PetscViewer viewer)
{
  PetscErrorCode  ierr;
  PetscBool       iascii;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Thread Communicator\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Number of threads = %D\n",tcomm->ncommthreads);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Type = %s\n",tcomm->pool->type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    if (tcomm->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*tcomm->ops->view)(tcomm,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetNThreads"
/*@C
   PetscThreadCommGetNThreads - Gets the thread count from the thread communicator
                                associated with the MPI communicator

   Not collective

   Input Parameters:
.  comm - the MPI communicator

   Output Parameters:
.  nthreads - number of threads

   Level: developer

   Notes:
   Returns -1 if threadcomm has not been created yet.

.seealso: PetscThreadCommSetNThreads()
@*/
PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm comm,PetscInt *nthreads)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (tcomm) {
    *nthreads = tcomm->ncommthreads;
  } else {
    *nthreads = -1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetAffinities"
/*@C
   PetscThreadCommGetAffinities - Returns the core affinities set for the
                                  thread communicator associated with the MPI_Comm

    Not collective

    Input Parameters:
.   comm - MPI communicator

    Output Parameters:
.   affinities - thread affinities

    Level: developer

    Notes:
    The user must allocate space (nthreads PetscInts) for the
    affinities. Must call PetscThreadCommSetAffinities before.

*/
PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm comm,PetscInt affinities[])
{
  PetscInt        i;
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  PetscValidIntPointer(affinities,2);
  for (i=0; i<tcomm->ncommthreads; i++) {
    affinities[i] = tcomm->commthreads[i]->affinity;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier"
/*@
   PetscThreadCommBarrier - Calls a job barrier on the threadcomm associated
                            with the MPI communicator

   Not Collective

   Input Parameters:
.  comm - MPI Communicator with attached threadcomm

   Level: developer

   Notes:
   Extracts threadcomm from the MPI_Comm and calls job barrier
   to verify that the threadcomm has finished all of its jobs.
   Called by the master thread only.

@*/
PetscErrorCode PetscThreadCommBarrier(MPI_Comm comm)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommJobBarrier(tcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommUserBarrier"
/*  PetscThreadUserCommBarrier - Calls a barrier on the thread communicator
                                 associated with the MPI communicator

    Collective on Threadcomm

    Input Parameters:
.   comm - the MPI communicator

    Level: developer

    Notes:
    Extracts threadcomm from the MPI_Comm and calls a barrier to
    verify that the all threads in the threadcomm have reached this point in
    the code.
    Must be called by all threads in the threadcomm.

*/
PetscErrorCode PetscThreadCommUserBarrier(MPI_Comm comm)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  ierr = (tcomm->ops->barrier)(tcomm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommJobBarrier"
/*
   PetscThreadCommJobBarrier - Barrier to verify that all threads in a
                               threadcomm have finished all of their jobs

   Not Collective

   Input Parameters:
.  tcomm - Threadcomm

   Level: developer

   Notes: Loops until all threads have finished the last job they were given.
   Called by master thread while worker threads are in threadpool spin loop.

*/
PetscErrorCode PetscThreadCommJobBarrier(PetscThreadComm tcomm)
{
  PetscInt                active_threads=0,i;
  PetscBool               wait          =PETSC_TRUE;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscInt                job_status;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  if (tcomm->ncommthreads == 1 && tcomm->ismainworker) PetscFunctionReturn(0);

  /* Loop till all threads signal that they have done their job */
  while (wait) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      jobqueue = tcomm->commthreads[i]->jobqueue;
      job = &jobqueue->jobs[jobqueue->newest_job_index];
      job_status      = job->job_status;
      active_threads += job_status;
    }
    if (PetscReadOnce(int,active_threads) > 0) active_threads = 0;
    else wait=PETSC_FALSE;
  }
  ierr = PetscLogEventEnd(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetScalars"
/*@C
   PetscThreadCommGetScalars - Gets pointers to locations for storing three PetscScalars that may be passed
                               to PetscThreadCommRunKernel to ensure that the scalar values remain valid
                               even after the main thread exits the calling function.

   Input Parameters:
+  comm - the MPI communicator having the thread communicator
.  val1 - pointer to store the first scalar value
.  val2 - pointer to store the second scalar value
-  val3 - pointer to store the third scalar value

   Level: developer

   Notes:
   This is a utility function to ensure that any scalars passed to PetscThreadCommRunKernel remain
   valid even after the main thread exits the calling function. If any scalars need to passed to
   PetscThreadCommRunKernel then these should be first stored in the locations provided by PetscThreadCommGetScalars()

   Pass NULL if any pointers are not needed.

   Called by the main thread only, not from within kernels

   Typical usage:

   PetscScalar *valptr;
   PetscThreadCommGetScalars(comm,&valptr,PETSC_NULL,PETSC_NULL);
   *valptr = alpha;   (alpha is the scalar you wish to pass in PetscThreadCommRunKernel)

   PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,valptr);

.seealso: PetscThreadCommRunKernel()
@*/
PetscErrorCode PetscThreadCommGetScalars(MPI_Comm comm,PetscScalar **val1, PetscScalar **val2, PetscScalar **val3)
{
  PetscInt                i;
  PetscErrorCode          ierr;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  for (i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    job      = &jobqueue->jobs[jobqueue->next_job_index];
    if (val1) *val1 = &job->scalars[0];
    if (val2) *val2 = &job->scalars[1];
    if (val3) *val3 = &job->scalars[2];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetInts"
/*@C
   PetscThreadCommGetInts - Gets pointers to locations for storing three PetscInts that may be passed
                               to PetscThreadCommRunKernel to ensure that the scalar values remain valid
                               even after the main thread exits the calling function.

   Input Parameters:
+  comm - the MPI communicator having the thread communicator
.  val1 - pointer to store the first integer value
.  val2 - pointer to store the second integer value
-  val3 - pointer to store the third integer value

   Level: developer

   Notes:
   This is a utility function to ensure that any scalars passed to PetscThreadCommRunKernel remain
   valid even after the main thread exits the calling function. If any scalars need to passed to
   PetscThreadCommRunKernel then these should be first stored in the locations provided by PetscThreadCommGetInts()

   Pass NULL if any pointers are not needed.

   Called by the main thread only, not from within kernels

   Typical usage:

   PetscScalar *valptr;
   PetscThreadCommGetScalars(comm,&valptr,PETSC_NULL,PETSC_NULL);
   *valptr = alpha;   (alpha is the scalar you wish to pass in PetscThreadCommRunKernel)

   PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,valptr);

.seealso: PetscThreadCommRunKernel()
@*/
PetscErrorCode PetscThreadCommGetInts(MPI_Comm comm,PetscInt **val1, PetscInt **val2, PetscInt **val3)
{
  PetscInt                i;
  PetscErrorCode          ierr;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  ierr    = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    job      = &jobqueue->jobs[jobqueue->next_job_index];
    if (val1) *val1 = &job->ints[0];
    if (val2) *val2 = &job->ints[1];
    if (val3) *val3 = &job->ints[2];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel"
/*@C
   PetscThreadCommRunKernel - Runs the kernel using the thread communicator
                              associated with the MPI communicator

   Not Collective

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  nargs - Number of input arguments for the kernel
-  ...   - variable list of input arguments

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,z);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x, PetscScalar* y, PetscReal* z)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),PetscInt nargs,...)
{
  PetscErrorCode          ierr;
  va_list                 argptr;
  PetscInt                i,j;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  if (nargs > PETSC_KERNEL_NARGS_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Requested %D input arguments for kernel, max. limit %D",nargs,PETSC_KERNEL_NARGS_MAX);
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);

  // Set job information for all threads
  for (i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    job      = &jobqueue->jobs[jobqueue->next_job_index];
    // Make sure previous job completed
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    // Prepare to run kernel by initializing next job
    job->tcomm          = tcomm;
    job->commrank       = i;
    job->nargs          = nargs;
    job->pfunc          = (PetscThreadKernel)func;
    va_start(argptr,nargs);
    for (j=0; j<nargs; j++) job->args[j] = va_arg(argptr,void*);
    va_end(argptr);
    job->job_status            = THREAD_JOB_POSTED;
    jobqueue->newest_job_index = jobqueue->next_job_index;
    jobqueue->next_job_index   = (jobqueue->next_job_index+1)%tcomm->nkernels;
    jobqueue->total_jobs_ctr++;
  }

  // Run Kernel for main thread
  jobqueue = tcomm->commthreads[0]->jobqueue;
  job      = &jobqueue->jobs[jobqueue->newest_job_index];
  if (tcomm->threadtype==THREAD_TYPE_NOTHREAD) {
    ierr = PetscRunKernel(0,job->nargs,job);CHKERRQ(ierr);
    job->job_status = THREAD_JOB_COMPLETED;
  } else {
    ierr = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel0_Private"
/*
   PetscThreadCommRunKernel0_Private - The zero-argument kernel needs to be callable with an
                                       unwrapped PetscThreadComm after ThreadComm keyval has
                                       been freed.

   Not Collective

   Input Parameters:
+  tcomm - Thread communicator
-  func  - the kernel (needs to be cast to PetscThreadKernel)

   Level: developer

   Notes:
   Runs kernel taking threadcomm as input instead of a MPI communicator.

*/
static PetscErrorCode PetscThreadCommRunKernel0_Private(PetscThreadComm tcomm,PetscErrorCode (*func)(PetscInt,...))
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  // Run kernel for nothread thread type
  if (tcomm->threadtype == THREAD_TYPE_NOTHREAD) {
    ierr = (*func)(0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // Set job information for all threads
  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;

    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->next_job_index];
    // Make sure previous job completed
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    // Prepare to run kernel
    job->tcomm                 = tcomm;
    job->commrank              = i;
    job->nargs                 = 1;
    job->pfunc                 = (PetscThreadKernel)func;
    job->job_status            = THREAD_JOB_POSTED;
    jobqueue->newest_job_index = jobqueue->next_job_index;
    jobqueue->next_job_index   = (jobqueue->next_job_index+1)%tcomm->nkernels;
    jobqueue->total_jobs_ctr++;
  }

  // Run kernel for main thread
  jobqueue = tcomm->commthreads[0]->jobqueue;
  job      = &jobqueue->jobs[jobqueue->newest_job_index];
  ierr     = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel0"
/*@C
   PetscThreadCommRunKernel0 - PetscThreadCommRunKernel version for kernels with no
                               input arguments

   Input Parameters:
+  comm  - the MPI communicator
-  func  - the kernel (needs to be cast to PetscThreadKernel)

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel0(comm,(PetscThreadKernel)kernel_func);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel0(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...))
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel0_Private(tcomm,func);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel1"
/*@C
   PetscThreadCommRunKernel1 - PetscThreadCommRunKernel version for kernels with 1
                               input argument

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
-  in1   - input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel1(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);

  // Run kernel for nothread thread type
  if (tcomm->threadtype == THREAD_TYPE_NOTHREAD) {
    ierr = (*func)(0,in1);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // Set job information for all threads
  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;

    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->next_job_index];
    // Make sure previous job completed
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    // Prepare to run kernel
    job->tcomm                 = tcomm;
    job->commrank              = i;
    job->nargs                 = 1;
    job->pfunc                 = (PetscThreadKernel)func;
    job->args[0]               = in1;
    job->job_status            = THREAD_JOB_POSTED;
    jobqueue->newest_job_index = jobqueue->next_job_index;
    jobqueue->next_job_index   = (jobqueue->next_job_index+1)%tcomm->nkernels;
    jobqueue->total_jobs_ctr++;
  }

  // Run kernel for main thread
  jobqueue = tcomm->commthreads[0]->jobqueue;
  job      = &jobqueue->jobs[jobqueue->newest_job_index];
  ierr     = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel2"
/*@C
   PetscThreadCommRunKernel2 - PetscThreadCommRunKernel version for kernels with 2
                               input arguments

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - 1st input argument for the kernel
-  in2   - 2nd input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt *x,PetscInt *y)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel2(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);

  // Run kernel for nothread thread type
  if (tcomm->threadtype == THREAD_TYPE_NOTHREAD) {
    ierr = (*func)(0,in1,in2);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // Set job information for all threads
  for (i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->next_job_index];
    // Make sure previous job completed
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    // Prepare to run kernel
    job->tcomm                 = tcomm;
    job->commrank              = i;
    job->nargs                 = 2;
    job->pfunc                 = (PetscThreadKernel)func;
    job->args[0]               = in1;
    job->args[1]               = in2;
    job->job_status            = THREAD_JOB_POSTED;
    jobqueue->newest_job_index = jobqueue->next_job_index;
    jobqueue->next_job_index   = (jobqueue->next_job_index+1)%tcomm->nkernels;
    jobqueue->total_jobs_ctr++;
  }

  // Run kernel for main thread
  jobqueue = tcomm->commthreads[0]->jobqueue;
  job      = &jobqueue->jobs[jobqueue->newest_job_index];
  ierr     = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel3"
/*@C
   PetscThreadCommRunKernel3 - PetscThreadCommRunKernel version for kernels with 3
                               input argument

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - first input argument for the kernel
.  in2   - second input argument for the kernel
-  in3   - third input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel3(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2,void *in3)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);

  // Run kernel for nothread thread type
  if (tcomm->threadtype == THREAD_TYPE_NOTHREAD) {
    ierr = (*func)(0,in1,in2,in3);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // Set job information for all threads
  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->next_job_index];
    // Make sure previous job completed
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    // Prepare to run kernel
    job->tcomm                 = tcomm;
    job->commrank              = i;
    job->nargs                 = 3;
    job->pfunc                 = (PetscThreadKernel)func;
    job->args[0]               = in1;
    job->args[1]               = in2;
    job->args[2]               = in3;
    job->job_status            = THREAD_JOB_POSTED;
    jobqueue->newest_job_index = jobqueue->next_job_index;
    jobqueue->next_job_index   = (jobqueue->next_job_index+1)%tcomm->nkernels;
    jobqueue->total_jobs_ctr++;
  }

  // Run kernel for main thread
  jobqueue = tcomm->commthreads[0]->jobqueue;
  job      = &jobqueue->jobs[jobqueue->newest_job_index];
  ierr     = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel4"
/*@C
   PetscThreadCommRunKernel4 - PetscThreadCommRunKernel version for kernels with 4
                               input argument

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - first input argument for the kernel
.  in2   - second input argument for the kernel
.  in3   - third input argument for the kernel
-  in4   - fourth input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel4(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2,void *in3,void *in4)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);

  // Run kernel for nothread thread type
  if (tcomm->threadtype==THREAD_TYPE_NOTHREAD) {
    ierr = (*func)(0,in1,in2,in3,in4);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // Set job information for all threads
  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->next_job_index];
    // Make sure previous job completed
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    // Prepare to run kernel
    job->tcomm                 = tcomm;
    job->commrank              = i;
    job->nargs                 = 4;
    job->pfunc                 = (PetscThreadKernel)func;
    job->args[0]               = in1;
    job->args[1]               = in2;
    job->args[2]               = in3;
    job->args[3]               = in4;
    job->job_status            = THREAD_JOB_POSTED;
    jobqueue->newest_job_index = jobqueue->next_job_index;
    jobqueue->next_job_index   = (jobqueue->next_job_index+1)%tcomm->nkernels;
    jobqueue->total_jobs_ctr++;
  }

  // Run kernel for main thread
  jobqueue = tcomm->commthreads[0]->jobqueue;
  job      = &jobqueue->jobs[jobqueue->newest_job_index];
  ierr     = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel6"
/*@C
   PetscThreadCommRunKernel6 - PetscThreadCommRunKernel version for kernels with 6
                               input arguments

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - first input argument for the kernel
.  in2   - second input argument for the kernel
.  in3   - third input argument for the kernel
.  in4   - fourth input argument for the kernel
.  in5   - fifth input argument for the kernel
-  in6   - sixth input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel6(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2,void *in3,void *in4,void *in5,void *in6)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);

  // Run kernel for nothread thread type
  if (tcomm->threadtype == THREAD_TYPE_NOTHREAD) {
    ierr = (*func)(0,in1,in2,in3,in4,in5,in6);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // Set job information for all threads
  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->next_job_index];
    // Make sure previous job completed
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    // Prepare to run kernel
    job->tcomm                 = tcomm;
    job->commrank              = i;
    job->nargs                 = 6;
    job->pfunc                 = (PetscThreadKernel)func;
    job->args[0]               = in1;
    job->args[1]               = in2;
    job->args[2]               = in3;
    job->args[3]               = in4;
    job->args[4]               = in5;
    job->args[5]               = in6;
    job->job_status            = THREAD_JOB_POSTED;
    jobqueue->newest_job_index = jobqueue->next_job_index;
    jobqueue->next_job_index   = (jobqueue->next_job_index+1)%tcomm->nkernels;
    jobqueue->total_jobs_ctr++;
  }

  // Run kernel for main thread
  jobqueue = tcomm->commthreads[0]->jobqueue;
  job      = &jobqueue->jobs[jobqueue->newest_job_index];
  ierr     = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   PetscThreadCommDetach - Detaches the thread communicator from the MPI communicator

   Not Collective

   Input Parameters:
.  comm - MPI Communicator

   Level: developer

   Notes:
   If this MPI comm is a PETSc comm, delete the attached threadcomm.

*/
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDetach"
PetscErrorCode PetscThreadCommDetach(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MPI_Attr_delete(comm,Petsc_ThreadComm_keyval);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   PetscThreadCommAttach - Attach the thread communicator to the MPI communicator

   Not Collective

   Input Parameters:
+  comm  - MPI communicator
-  tcomm - Threadcomm to attach to MPI comm

   Level: developer

   Notes:
   If this MPI comm is a PETSc comm, attach the threadcomm.

*/
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommAttach"
PetscErrorCode PetscThreadCommAttach(MPI_Comm comm,PetscThreadComm tcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (!flg) {
    tcomm->refct++;
    ierr = MPI_Attr_put(comm,Petsc_ThreadComm_keyval,tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate"
/*@
   PetscThreadCommCreate - Create a thread communicator object

   Not Collective

   Input Parameters:
+  comm       - MPI communicator
.  nthreads   - Number of threads for communicator or PETSC_DECIDE
.  granks     - Array of ranks for each thread or PETSC_NULL
-  affinities - Array of core affinities for each thread or PETSC_NULL

   Output Parameters:
.  mpicomm    - New MPI_Comm with attached threadcomm

   Level: developer

   Notes:
   Allocates and initializes a threadcomm. Duplicates the input MPI comm and attaches the
   created threadcomm to it. Creates a new threadpool for the threadcomm to use.
   PETSc can automatically set nthreads, granks, and affinities. The granks and affinities
   arrays can be the same array.

@*/
PetscErrorCode PetscThreadCommCreate(MPI_Comm comm,PetscInt nthreads,PetscInt *granks,PetscInt *affinities,MPI_Comm *mpicomm)
{
  PetscInt        i,*lranks;
  PetscThreadComm tcomm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  // Allocate space for ThreadComm
  ierr = PetscThreadCommAlloc(&tcomm);CHKERRQ(ierr);
  // Create ThreadPool
  ierr = PetscThreadPoolCreate(tcomm,nthreads,granks,affinities);CHKERRQ(ierr);
  if (nthreads == PETSC_DECIDE) {
    nthreads = tcomm->pool->npoolthreads;
  }
  // Set thread ranks
  ierr = PetscMalloc1(nthreads,&lranks);CHKERRQ(ierr);
  for (i=0; i<nthreads; i++) {
    lranks[i] = i;
  }
  // Initialize ThreadComm
  ierr = PetscThreadCommInitialize(nthreads,lranks,tcomm);CHKERRQ(ierr);
  // Duplicate MPI_Comm
  ierr = PetscCommForceDuplicate(comm,mpicomm,PETSC_NULL);CHKERRQ(ierr);
  // Attach ThreadComm to new MPI_Comm
  ierr = PetscThreadCommAttach(*mpicomm,tcomm);CHKERRQ(ierr);
  printf("Created new threadcomm with %d threads\n",tcomm->ncommthreads);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreateShare"
/*@
   PetscThreadCommCreateShare - Create a thread communicator object that uses the threadpool
                                from another thread communicator

   Not Collective

   Input Parameters:
+  comm     - MPI communicator
.  nthreads - Number of threads for communicator or PETSC_DECIDE
-  granks   - Array of ranks for each thread or PETSC_NULL

   Output Parameters:
.  mpicomm  - New MPI_Comm with attached threadcomm

   Level: developer

   Notes:
   Allocates and initializes a threadcomm. Duplicates the input MPI comm and attaches the
   created threadcomm to it. Uses the threadcomm from the input mpi comm.
   If user passes in PETSC_DECIDE for nthreads then use all threads in the original threadcomm.
   If user passes in PETSC_NULL for granks then use the first nthreads threads in the original threadcomm.

@*/
PetscErrorCode PetscThreadCommCreateShare(MPI_Comm comm,PetscInt nthreads,PetscInt *granks,MPI_Comm *mpicomm)
{
  PetscThreadComm tcomm,incomm;
  PetscInt        i,j,*lranks;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  printf("Creating new comm with %d threads\n",nthreads);
  // Get input threadcomm
  ierr = PetscCommGetThreadComm(comm,&incomm);CHKERRQ(ierr);
  // Make sure input arguments are reasonable
  if (nthreads > incomm->ncommthreads) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot created a shared threadcomm with more threads than original threadcomm");
  }
  // Allocate space for new threadcomm
  ierr = PetscThreadCommAlloc(&tcomm);CHKERRQ(ierr);
  // Set new threadcomm to use input threadpool
  tcomm->pool = incomm->pool;
  tcomm->pool->refct++;
  // If user did not pass in nthreads, set nthreads equal to nthreads in the input threadcomm
  if (nthreads == PETSC_DECIDE) {
    nthreads = incomm->ncommthreads;
  }
  // If user did not pass in granks, use the first nthreads threads in the incomm
  if (granks == PETSC_NULL) {
    for (i=0; i<nthreads; i++) {
      granks[i] = incomm->commthreads[i]->grank;
    }
  }
  // Convert granks to lranks
  ierr = PetscMalloc1(nthreads,&lranks);CHKERRQ(ierr);
  for(i=0; i<nthreads; i++) {
    lranks[i] = -1;
    for(j=0; j<tcomm->pool->npoolthreads; j++) {
      if(granks[i]==tcomm->pool->poolthreads[j]->grank) lranks[i] = j;
    }
    if (lranks[i]==-1) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Original threadcomm does not contain requested thread for shared threadcomm");
    }
  }
  // Initialize ThreadComm
  ierr = PetscThreadCommInitialize(nthreads,lranks,tcomm);CHKERRQ(ierr);
  // Duplicate MPI_Comm
  ierr = PetscCommForceDuplicate(comm,mpicomm,PETSC_NULL);CHKERRQ(ierr);
  // Remove original threadcomm
  ierr = PetscThreadCommDetach(*mpicomm);CHKERRQ(ierr);
  // Attach ThreadComm to new MPI_Comm
  ierr = PetscThreadCommAttach(*mpicomm,tcomm);CHKERRQ(ierr);
  printf("Created new threadcomm with %d threads\n",tcomm->ncommthreads);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreateAttach"
/*@
   PetscThreadCommCreateAttach - Create a thread communicator object and attach it to the MPI Comm

   Not Collective

   Input Parameters:
+  comm       - MPI communicator
.  nthreads   - Number of threads for communicator or PETSC_DECIDE
.  granks     - Array of ranks for each thread or PETSC_NULL
-  affinities - Array of core affinities for each thread or PETSC_NULL

   Level: developer

   Notes:
   Allocates and initializes a threadcomm. Attaches the created threadcomm to the input
   MPI Comm. Creates a new threadpool for the threadcomm to use.
   PETSc can automatically set nthreads, granks, and affinities. The granks and affinities
   arrays can be the same array.

@*/
PetscErrorCode PetscThreadCommCreateAttach(MPI_Comm comm,PetscInt nthreads,PetscInt *granks,PetscInt *affinities)
{
  PetscInt        i,*lranks;
  PetscThreadComm tcomm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  printf("Calling PetscThreadCommCreateAttach\n");
  // Allocate space for ThreadComm
  ierr = PetscThreadCommAlloc(&tcomm);CHKERRQ(ierr);
  // Create ThreadPool
  ierr = PetscThreadPoolCreate(tcomm,nthreads,granks,affinities);CHKERRQ(ierr);
  if (nthreads == PETSC_DECIDE) {
    nthreads = tcomm->pool->npoolthreads;
  }
  // Set thread ranks
  ierr = PetscMalloc1(nthreads,&lranks);CHKERRQ(ierr);
  for(i=0; i<nthreads; i++) {
    lranks[i] = i;
  }
  // Initialize ThreadComm
  ierr = PetscThreadCommInitialize(nthreads,lranks,tcomm);CHKERRQ(ierr);
  // Attach ThreadComm to MPI_Comm
  ierr = PetscThreadCommAttach(comm,tcomm);CHKERRQ(ierr);
  printf("Created new threadcomm with %d threads\n",tcomm->ncommthreads);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreateMultiple"
/*@
   PetscThreadCommCreateMultiple - Create multiple thread communicator objects and attach each
                                   to a duplicated MPI Comm.

   Not Collective

   Input Parameters:
+  comm        - MPI communicator
.  ncomms      - Number of thread comms to create
.  nthreads    - Number of threads in all communicators or PETSC_DECIDE
.  incommsizes - Array of sizes for each new comm to create or PETSC_NULL
.  granks      - Array of ranks for each thread or PETSC_NULL
-  affinities  - Array of core affinities for each thread or PETSC_NULL

   Output Parameters:
.  multcomms   - Array of MPI comms each with a different threadcomm

   Level: developer

   Notes:
   Allocates and initializes multiple threadcomms. Duplicates the input MPI comm and attaches
   a threadcomm to the duplicate MPI Comm. Creates a new threadpool for each threadcomm to use.
   PETSc can automatically set nthreads, incommsizes, granks, and affinities. The granks and
   affinities arrays can be the same array.

@*/
PetscErrorCode PetscThreadCommCreateMultiple(MPI_Comm comm,PetscInt ncomms,PetscInt nthreads,PetscInt *incommsizes,PetscInt *granks,PetscInt *affinities,MPI_Comm **multcomms)
{
  PetscErrorCode ierr;
  PetscInt       i,j,*multranks,*multaffinities,*commsizes;
  PetscInt       splitsize,remainder,nthr,startthread;
  PetscBool      extra,flg;

  PetscFunctionBegin;
  // Allocate splitcomms
  ierr = PetscMalloc1(ncomms,multcomms);CHKERRQ(ierr);

  // If user did not pass in commsizes, split them evenly
  if(!incommsizes) {

    // If user did not pass in nthreads, get from input options or use ncomms
    if (nthreads == PETSC_DECIDE) {
      nthreads = ncomms;
      ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Thread comm - setting number of threads",PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-threadcomm_nthreads","number of threads to use in the thread communicator","PetscThreadPoolSetNThreads",1,&nthr,&flg);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
      if (flg) {
        if (nthr == PETSC_DECIDE) nthreads = N_CORES;
        else nthreads = nthr;
      }
    }

    // Allocate and set commsizes
    ierr = PetscMalloc1(ncomms,&commsizes);CHKERRQ(ierr);

    splitsize = nthreads/ncomms;
    remainder = nthreads - splitsize*ncomms;
    for (i=0; i<ncomms; i++) {
      extra = (PetscBool)(i < remainder);
      commsizes[i] = extra ? splitsize+1 : splitsize;
    }
  } else {
    // Use input commsizes
    commsizes = incommsizes;
  }

  // Create each splitcomm
  for(i=0; i<ncomms; i++) {
    // Set ranks and affinities for threadcomm
    ierr = PetscMalloc1(commsizes[i],&multranks);CHKERRQ(ierr);
    ierr = PetscMalloc1(commsizes[i],&multaffinities);CHKERRQ(ierr);
    // Count previous threads
    startthread = 0;
    for(j=0; j<i; j++) {
      startthread += commsizes[j];
    }
    // Set default affinities this multcomm will use
    for(j=0; j<commsizes[i]; j++) {
      if(!granks) multranks[j] = startthread + j;
      else multranks[j] = granks[j];
      if(!affinities) multaffinities[j] = startthread + j;
      else multaffinities[j] = affinities[j];
      printf("Creating multcomm thread=%d rank=%d on core=%d\n",j,multranks[j],multaffinities[j]);
    }
    // Create threadcomm
    ierr = PetscThreadCommCreate(comm,commsizes[i],multranks,multaffinities,&(*multcomms)[i]);CHKERRQ(ierr);

    ierr = PetscFree(multranks);CHKERRQ(ierr);
    ierr = PetscFree(multaffinities);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSplit"
/*@
   PetscThreadCommCreateSplit - Split a threadcomm into multiple smaller threadcomms, with each
                                using the same threadpool as the original threadcomm.

   Not Collective

   Input Parameters:
+  comm        - MPI communicator
.  ncomms      - Number of thread comms to create
.  incommsizes - Array of sizes for each new comm to create or PETSC_NULL
-  granks      - Array of global ranks for each thread in splitcomm or PETSC_NULL

   Output Parameters:
.  multcomms   - Array of MPI comms each with a different threadcomm

   Level: developer

   Notes:
   Allocates and initializes multiple threadcomms. Duplicates the input MPI comm and attaches
   a threadcomm to the duplicate MPI Comm. Creates a new threadpool for each threadcomm to use.
   PETSc can automatically set nthreads, incommsizes, granks, and affinities. The granks and
   affinities arrays can be the same array.

@*/
PetscErrorCode PetscThreadCommSplit(MPI_Comm comm,PetscInt ncomms,PetscInt *incommsizes,PetscInt *ingranks,MPI_Comm **splitcomms)
{
  PetscErrorCode  ierr;
  PetscInt        i,j,*granks,startthread,*commsizes,splitsize,remainder,nthreads;
  PetscBool       extra;
  PetscThreadComm tcomm;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);

  // Allocate splitcomms
  ierr = PetscMalloc1(ncomms,splitcomms);

  // Check that split threadcomms use same number of threads as original threadcomm
  if (incommsizes) {
    for (i=0; i<ncomms; i++) {
      nthreads += incommsizes[i];
    }
  }
  if (nthreads > tcomm->ncommthreads) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Splitcomm contains more threads than original comm");

  // If user did not pass in commsizes split them evenly
  if (!incommsizes) {
    ierr = PetscMalloc1(ncomms,&commsizes);CHKERRQ(ierr);
    splitsize = tcomm->ncommthreads/ncomms;
    remainder = tcomm->ncommthreads - splitsize*ncomms;
    for (i=0; i<ncomms; i++) {
      extra = (PetscBool)(i < remainder);
      commsizes[i] = extra ? splitsize+1 : splitsize;
    }
  } else {
    commsizes = incommsizes;
  }

  // Create each splitcomm
  for (i=0; i<ncomms; i++) {
    PetscMalloc1(commsizes[i],&granks);
    // Count previous threads
    startthread = 0;
    for (j=0; j<i; j++) {
      startthread += commsizes[j];
    }
    // Set granks
    if (!ingranks) {
      for (j=0; j<commsizes[i]; j++) {
        granks[j] = startthread + j;
        printf("Creating splitcomm %d with thread %d\n",j,granks[j]);
      }
    } else {
      for (j=0; j<commsizes[i]; j++) {
        granks[j] = ingranks[startthread+j];
      }
    }
    // Create threadcomm that shares threads with input threadcomm
    ierr = PetscThreadCommCreateShare(comm,commsizes[i],granks,&(*splitcomms)[i]);CHKERRQ(ierr);
    ierr = PetscFree(granks);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitialize"
/*
   PetscThreadCommInitialize - Initializes a thread communicator object

   Not Collective

   Input Parameters:
+  nthreads - Number of threads for the threadcomm
.  lranks   - Threadpool rank for each thread to add to the threadcomm
-  tcomm    - Threadcomm to initialize

   Level: developer

   Notes:
   Allocates the threadcomm based on the settings from the threadpool. The threadcomm
   gains access to the requested threads in the threadpool.
   Defaults to using the nonthreaded communicator.
*/
PetscErrorCode PetscThreadCommInitialize(PetscInt nthreads,PetscInt *lranks,PetscThreadComm tcomm)
{
  PetscThreadPool pool;
  PetscInt        i;
  PetscErrorCode  ierr;
  PetscBool       flg;

  PetscFunctionBegin;
  printf("In threadcomm_init with %d threads\n",nthreads);
  pool = tcomm->pool;

  // Get option settings from command line
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Threadcomm options",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-threadcomm_syncafter","Puts a barrier after every kernel call",PETSC_NULL,PETSC_TRUE,&tcomm->syncafter,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  tcomm->model = pool->model;
  tcomm->threadtype = pool->threadtype;
  tcomm->nkernels = pool->nkernels;
  printf("init based on model=%d\n",tcomm->model);
  if (tcomm->model==THREAD_MODEL_LOOP) {
    tcomm->ismainworker = PETSC_TRUE;
    tcomm->thread_start = 1;
    tcomm->ncommthreads = nthreads;
  } else if (tcomm->model==THREAD_MODEL_AUTO) {
    tcomm->ismainworker = PETSC_FALSE;
    tcomm->thread_start = 0;
    tcomm->ncommthreads = nthreads;
  } else if (tcomm->model==THREAD_MODEL_USER) {
    tcomm->ismainworker = PETSC_TRUE;
    tcomm->thread_start = 1;
    tcomm->ncommthreads = nthreads;
  }
  printf("threadstart=%d ncommthreads=%d\n",tcomm->thread_start,tcomm->ncommthreads);

  ierr = PetscMalloc1(tcomm->ncommthreads,&tcomm->commthreads);
  for(i=0; i<tcomm->ncommthreads; i++) {
    printf("threadcomm thread i=%d gets poolthreads lranks=%d\n",i,lranks[i]);
    tcomm->commthreads[i] = pool->poolthreads[lranks[i]];
  }

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  // Set affinity for main thread
  if (tcomm->ismainworker) {
    PetscBool set;
    cpu_set_t cpuset;
    printf("main thread setting affinity to grank=%d\n",tcomm->commthreads[0]->grank);
    PetscThreadPoolSetAffinity(tcomm->pool,&cpuset,tcomm->commthreads[0]->grank,&set);
    sched_setaffinity(0,sizeof(cpu_set_t),&cpuset);
  }
#endif

  /* Set the leader thread rank */
  if (tcomm->ncommthreads) {
    tcomm->lleader = 0;
    tcomm->gleader = tcomm->commthreads[0]->grank;
  }

  /* Initialize implementation specific settings */
  printf("Initializing threadcomm implementation settings\n");
  ierr = (*pool->ops->tcomminit)(tcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommReductionCreate(tcomm,&tcomm->red);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetOwnershipRanges"
/*
   PetscThreadCommGetOwnershipRanges - Given the global size of an array, compute the local sizes
                                       and sets the starting array indices

   Input Parameters:
+  comm - the MPI communicator which holds the thread communicator
-  N    - the global size of the array

   Output Parameters:
.  trstarts - The starting array indices for each thread. the size of trstarts is nthreads+1

   Notes:
   trstarts is malloced in this routine
*/
PetscErrorCode PetscThreadCommGetOwnershipRanges(MPI_Comm comm,PetscInt N,PetscInt *trstarts[])
{
  PetscErrorCode  ierr;
  PetscInt        Q,R;
  PetscBool       S;
  PetscThreadComm tcomm = PETSC_NULL;
  PetscInt        *trstarts_out,nloc,i;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);

  ierr            = PetscMalloc1((tcomm->ncommthreads+1),&trstarts_out);CHKERRQ(ierr);
  trstarts_out[0] = 0;
  Q               = N/tcomm->ncommthreads;
  R               = N - Q*tcomm->ncommthreads;
  for (i=0; i<tcomm->ncommthreads; i++) {
    S                 = (PetscBool)(i < R);
    nloc              = S ? Q+1 : Q;
    trstarts_out[i+1] = trstarts_out[i] + nloc;
  }

  *trstarts = trstarts_out;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetRank"
/*
   PetscThreadCommGetRank - Gets the rank of the calling thread

   Input Parameters:
.  tcomm - the thread communicator

   Output Parameters:
.  trank - The rank of the calling thread

*/
PetscErrorCode PetscThreadCommGetRank(PetscThreadComm tcomm,PetscInt *trank)
{
  PetscErrorCode ierr;
  PetscInt       rank = 0;

  PetscFunctionBegin;
  if (tcomm->ops->getrank) {
    ierr = (*tcomm->ops->getrank)(&rank);CHKERRQ(ierr);
  }
  *trank = rank;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitTypeRegister"
/*@C
  PetscThreadCommTypeRegister -

  Level: advanced
@*/
PetscErrorCode PetscThreadCommInitTypeRegister(const char sname[],PetscErrorCode (*function)(PetscThreadPool))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscThreadCommInitTypeList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommTypeRegister"
/*@C
  PetscThreadCommTypeRegister -

  Level: advanced
@*/
PetscErrorCode  PetscThreadCommTypeRegister(const char sname[],PetscErrorCode (*function)(PetscThreadComm))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscThreadCommTypeList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommModelRegister"
/*@C
  PetscThreadCommModelRegister -

  Level: advanced
@*/
PetscErrorCode  PetscThreadCommModelRegister(const char sname[],PetscErrorCode (*function)(PetscThreadComm))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscThreadCommModelList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreateModel_Loop"
PetscErrorCode PetscThreadCommCreateModel_Loop(PetscThreadPool pool)
{
  PetscFunctionBegin;
  printf("Creating Loop Model\n");
  pool->model = THREAD_MODEL_LOOP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreateModel_Auto"
PetscErrorCode PetscThreadCommCreateModel_Auto(PetscThreadPool pool)
{
  PetscFunctionBegin;
  printf("Creating Auto Model\n");
  pool->model = THREAD_MODEL_AUTO;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreateModel_User"
PetscErrorCode PetscThreadCommCreateModel_User(PetscThreadPool pool)
{
  PetscFunctionBegin;
  printf("Creating User Model\n");
  pool->model = THREAD_MODEL_USER;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommJoin"
PetscErrorCode PetscThreadCommJoin(MPI_Comm *comm,PetscInt ncomms,PetscInt trank,PetscInt *commrank)
{
  PetscInt i, j;
  PetscThreadComm *tcomm;
  PetscErrorCode ierr;
  PetscInt comm_index=-1, local_index=-1, startthread=0;

  PetscFunctionBegin;

  printf("rank=%d joining thread pool\n",trank);
  PetscMalloc1(ncomms,&tcomm);

  // Determine which threadcomm and local thread this thread belongs to
  for(i=0; i<ncomms; i++) {
    ierr = PetscCommGetThreadComm(comm[i],&tcomm[i]);

    // Check if this thread is in threadcomm
    for(j=0; j<tcomm[i]->ncommthreads; j++) {
      if(tcomm[i]->commthreads[j]->grank==trank) {
        comm_index = i;
        local_index = j;
      }
    }
    startthread += tcomm[i]->ncommthreads;
  }

  // Make sure this thread is in a comm
  if(comm_index>=0) {
    printf("trank=%d comm_index=%d local_index=%d lleader=%d\n",trank,comm_index,local_index,tcomm[comm_index]->lleader);

    // Make sure all threads have reached this routine
    ierr = (*tcomm[comm_index]->ops->barrier)(tcomm[comm_index]);

    // Initialize thread and join threadpool if a worker thread
    if(local_index==tcomm[comm_index]->lleader) {
      tcomm[comm_index]->active = PETSC_TRUE;
      *commrank = comm_index;
    } else {
      tcomm[comm_index]->commthreads[local_index]->status = THREAD_INITIALIZED;
      tcomm[comm_index]->commthreads[local_index]->jobdata = 0;
      tcomm[comm_index]->commthreads[local_index]->pool = tcomm[comm_index]->pool;
      *commrank = -1;
    }

    // Make sure all threads have initialized threadcomm
    ierr = (*tcomm[comm_index]->ops->barrier)(tcomm[comm_index]);

    if(*commrank==-1) {
      // Join thread pool if not leader thread
      PetscThreadPoolFunc((void*)&tcomm[comm_index]->commthreads[local_index]);
    }
  } else {
    *commrank = -1;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommReturn"
PetscErrorCode PetscThreadCommReturn(MPI_Comm *comm,PetscInt ncomms,PetscInt trank,PetscInt *commrank)
{
  PetscThreadComm *tcomm;
  PetscInt i, j, comm_index=-1, startthread=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscMalloc1(ncomms,&tcomm);
  // Determine which threadcomm and local thread this thread belongs to
  for(i=0; i<ncomms; i++) {
    ierr = PetscCommGetThreadComm(comm[i],&tcomm[i]);

    // Check if this thread is in threadcomm
    for(j=0; j<tcomm[i]->ncommthreads; j++) {
      if(tcomm[i]->commthreads[j]->grank==trank) {
        comm_index = i;
      }
    }
    startthread += tcomm[i]->ncommthreads;
  }

  // Make sure this thread is in a comm
  if(comm_index>=0) {
    // Master threads terminate worker threads
    if(*commrank>=0) {
      printf("Returning all threads\n");
      ierr = PetscThreadCommJobBarrier(tcomm[comm_index]);
      for(i=0; i<tcomm[comm_index]->ncommthreads; i++) {
        tcomm[comm_index]->commthreads[i]->status = THREAD_TERMINATE;
      }
    }

    // Make sure all threads have reached this point
    ierr = (*tcomm[comm_index]->ops->barrier)(tcomm[comm_index]);
  }
  *commrank = -1;
  PetscFunctionReturn(0);
}
