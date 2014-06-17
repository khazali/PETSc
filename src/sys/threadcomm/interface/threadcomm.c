#include <petsc-private/threadcommimpl.h>      /*I "petscthreadcomm.h" I*/
#include <petscviewer.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

PetscThreadComm   PETSC_THREAD_COMM_WORLD = NULL;

/* Logging support */
PetscLogEvent ThreadComm_RunKernel, ThreadComm_Barrier;

static PetscErrorCode PetscThreadCommRunKernel0_Private(PetscThreadComm tcomm,PetscErrorCode (*func)(PetscInt,...));

PetscErrorCode PetscThreadCommWorldInitialize();
#undef __FUNCT__
#define __FUNCT__ "PetscGetThreadCommWorld"
/*
  PetscGetThreadCommWorld - Gets the global thread communicator.
                            Creates it if it does not exist already.

  Not Collective

  Output Parameters:
  tcommp - pointer to the global thread communicator

  Level: Intermediate
*/
PetscErrorCode PetscGetThreadCommWorld(PetscThreadComm *tcommp)
{
  PetscThreadPool pool;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!PETSC_THREAD_COMM_WORLD) {
    // Make sure thread pool is created
    ierr = PetscThreadPoolGetPool(PETSC_COMM_WORLD,&pool);
    // Create threadcomm world
    ierr = PetscThreadCommWorldInitialize();CHKERRQ(ierr);
  }
  *tcommp = PETSC_THREAD_COMM_WORLD;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCommGetThreadComm"
/*@C
  PetscCommGetThreadComm - Gets the thread communicator
                           associated with the MPI communicator

  Not Collective

  Input Parameters:
. comm - the MPI communicator

  Output Parameters:
. tcommp - pointer to the thread communicator

  Notes: If no thread communicator is on the MPI_Comm then the global thread communicator
         is returned.
  Level: Intermediate

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy()
@*/
PetscErrorCode PetscCommGetThreadComm(MPI_Comm comm,PetscThreadComm *tcomm)
{
  PetscThreadPool pool;
  PetscErrorCode  ierr;
  PetscInt        trank;
  PetscMPIInt     keyval,flg;
  void            *ptr;

  PetscFunctionBegin;
  ierr = PetscThreadPoolGetPool(comm,&pool);
  ierr = PetscThreadCommGetRank(&trank);
  keyval = pool->thread_keyvals[trank];
  ierr = MPI_Attr_get(comm,keyval,(PetscThreadComm*)&ptr,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscGetThreadCommWorld(tcomm);CHKERRQ(ierr);
  } else *tcomm = (PetscThreadComm)ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate"
/*
   PetscThreadCommCreate - Allocates a thread communicator object

   Not Collective

   Output Parameters:
.  tcomm - pointer to the thread communicator object

   Level: developer

.seealso: PetscThreadCommDestroy()
*/
PetscErrorCode PetscThreadCommCreate(PetscThreadComm *tcomm)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcommout;

  PetscFunctionBegin;
  PetscValidPointer(tcomm,2);

  *tcomm = NULL;
  ierr                   = PetscNew(&tcommout);CHKERRQ(ierr);

  tcommout->refct           = 0;
  tcommout->leader          = 0;
  tcommout->isnothread      = PETSC_TRUE;
  tcommout->thread_start    = 0;
  tcommout->red             = NULL;
  tcommout->active          = PETSC_FALSE;
  tcommout->keyval          = MPI_KEYVAL_INVALID;

  tcommout->ncommthreads    = -1;
  tcommout->nthreads        = 0;

  tcommout->jobqueue        = NULL;
  tcommout->job_ctr         = 0;
  tcommout->my_job_counter  = NULL;
  tcommout->my_kernel_ctr   = NULL;
  tcommout->glob_kernel_ctr = NULL;

  tcommout->barrier_threads = 0;
  tcommout->wait1           = PETSC_TRUE;
  tcommout->wait2           = PETSC_TRUE;

  *tcomm                    = tcommout;

  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)

static PetscErrorCode PetscThreadCommStackCreate_kernel(PetscInt trank)
{
  if (trank && !PetscStackActive()) {
    PetscStack *petscstack_in;
    petscstack_in = (PetscStack*)malloc(sizeof(PetscStack));
    petscstack_in->currentsize = 0;
    PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,petscstack_in);
  }
  return 0;
}

/* Creates stack frames for threads other than the main thread */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackCreate"
PetscErrorCode  PetscThreadCommStackCreate(void)
{
  PetscErrorCode ierr;
  ierr = PetscThreadCommRunKernel0(PETSC_COMM_SELF,(PetscThreadKernel)PetscThreadCommStackCreate_kernel);CHKERRQ(ierr);
  ierr = PetscThreadCommBarrier(PETSC_COMM_SELF);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode PetscThreadCommStackDestroy_kernel(PetscInt trank)
{
  if (trank && PetscStackActive()) {
    PetscStack *petscstack_in;
    petscstack_in = (PetscStack*)PetscThreadLocalGetValue(petscstack);
    free(petscstack_in);
    PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,(PetscStack*)0);
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackDestroy"
/* Destroy stack frames for threads other than main thread
 *
 * The keyval may have been destroyed by the time this function is called, thus we must call
 * PetscThreadCommRunKernel0_Private so that we never reference an MPI_Comm.
 */
PetscErrorCode  PetscThreadCommStackDestroy(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscThreadCommRunKernel0_Private(PETSC_THREAD_COMM_WORLD,(PetscThreadKernel)PetscThreadCommStackDestroy_kernel);CHKERRQ(ierr);
  PETSC_THREAD_COMM_WORLD = NULL;
  PetscFunctionReturn(0);
  return 0;
}
#else
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackCreate"
PetscErrorCode  PetscThreadCommStackCreate(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackDestroy"
PetscErrorCode  PetscThreadCommStackDestroy(void)
{
  PetscFunctionBegin;
  PETSC_THREAD_COMM_WORLD = NULL;
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy"
/*
  PetscThreadCommDestroy - Frees a thread communicator object

  Not Collective

  Input Parameters:
. tcomm - the PetscThreadComm object

  Level: developer

.seealso: PetscThreadCommCreate()
*/
PetscErrorCode PetscThreadCommDestroy(PetscThreadComm *tcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Destroying thread comm refct=%d\n",(*tcomm)->refct);
  if (!*tcomm) PetscFunctionReturn(0);
  if (!--(*tcomm)->refct) {
    //if(pool->model==THREAD_MODEL_LOOP) {
    //ierr = PetscThreadCommStackDestroy();CHKERRQ(ierr);
    //}

    ierr = MPI_Keyval_free(&(*tcomm)->keyval);CHKERRQ(ierr);
    ierr = PetscThreadCommReductionDestroy((*tcomm)->red);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm)->jobqueue->jobs[0].job_status);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm)->jobqueue->jobs);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm)->jobqueue->tinfo);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm)->jobqueue);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm));CHKERRQ(ierr);
  }
  *tcomm = NULL;
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
  PetscThreadPool pool = PETSC_THREAD_POOL;
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
    ierr = PetscViewerASCIIPrintf(viewer,"Type = %s\n",pool->type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    if (pool->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*pool->ops->view)(tcomm,viewer);CHKERRQ(ierr);
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

.seealso: PetscThreadCommSetNThreads()
@*/
PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm comm,PetscInt *nthreads)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr      = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  *nthreads = tcomm->ncommthreads;
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
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  PetscValidIntPointer(affinities,2);
  ierr = PetscMemcpy(affinities,pool->affinities,tcomm->ncommthreads*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier"
/*  PetscThreadCommBarrier - Apply a barrier on the thread communicator
                             associated with the MPI communicator

    Not collective

    Input Parameters:
.   comm - the MPI communicator

    Level: developer

    Notes:
    This routine provides an interface to put an explicit barrier between
    successive kernel calls to ensure that the first kernel is executed
    by all the threads before calling the next one.

    Called by the main thread only.

    May not be applicable to all types.
*/
PetscErrorCode PetscThreadCommBarrier(MPI_Comm comm)
{
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (pool->ops->kernelbarrier) {
    ierr = (*pool->ops->kernelbarrier)(tcomm);CHKERRQ(ierr);
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
   PetscThreadCommGetScalars(comm,&valptr,NULL,NULL);
   *valptr = alpha;   (alpha is the scalar you wish to pass in PetscThreadCommRunKernel)

   PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,valptr);

.seealso: PetscThreadCommRunKernel()
@*/
PetscErrorCode PetscThreadCommGetScalars(MPI_Comm comm,PetscScalar **val1, PetscScalar **val2, PetscScalar **val3)
{
  PetscErrorCode          ierr;
  PetscThreadComm         tcomm;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscInt                job_num;
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  ierr    = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  jobqueue = tcomm->jobqueue;
  job_num = jobqueue->ctr%pool->nkernels;
  job     = &jobqueue->jobs[job_num];
  if (val1) *val1 = &job->scalars[0];
  if (val2) *val2 = &job->scalars[1];
  if (val3) *val3 = &job->scalars[2];
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
   PetscThreadCommGetScalars(comm,&valptr,NULL,NULL);
   *valptr = alpha;   (alpha is the scalar you wish to pass in PetscThreadCommRunKernel)

   PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,valptr);

.seealso: PetscThreadCommRunKernel()
@*/
PetscErrorCode PetscThreadCommGetInts(MPI_Comm comm,PetscInt **val1, PetscInt **val2, PetscInt **val3)
{
  PetscErrorCode          ierr;
  PetscThreadComm         tcomm;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscInt                job_num;
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  ierr    = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  jobqueue = tcomm->jobqueue;
  job_num = jobqueue->ctr%pool->nkernels;
  job     = &jobqueue->jobs[job_num];
  if (val1) *val1 = &job->ints[0];
  if (val2) *val2 = &job->ints[1];
  if (val3) *val3 = &job->ints[2];
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
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  if (nargs > PETSC_KERNEL_NARGS_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Requested %D input arguments for kernel, max. limit %D",nargs,PETSC_KERNEL_NARGS_MAX);
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  jobqueue = tcomm->jobqueue;
  job  = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = jobqueue->ctr;
  job->nargs          = nargs;
  job->pfunc          = (PetscThreadKernel)func;
  va_start(argptr,nargs);
  for (i=0; i < nargs; i++) job->args[i] = va_arg(argptr,void*);
  va_end(argptr);
  for (i=0; i<tcomm->ncommthreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
  jobqueue->kernel_ctr++;
  if (tcomm->isnothread) {
    ierr = PetscRunKernel(0,job->nargs,job);CHKERRQ(ierr);
    job->job_status[0] = THREAD_JOB_COMPLETED;
  } else {
    ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel0_Private"
/* The zero-argument kernel needs to be callable with an unwrapped PetscThreadComm after ThreadComm keyval has been freed. */
static PetscErrorCode PetscThreadCommRunKernel0_Private(PetscThreadComm tcomm,PetscErrorCode (*func)(PetscInt,...))
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  if (tcomm->isnothread) {
    ierr = (*func)(0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  jobqueue = tcomm->jobqueue;

  if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = jobqueue->ctr;
  job->nargs          = 1;
  job->pfunc          = (PetscThreadKernel)func;

  for (i=0; i<tcomm->ncommthreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
  jobqueue->kernel_ctr++;

  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);
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
  PetscErrorCode        ierr;
  PetscThreadComm       tcomm=0;

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
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  jobqueue = tcomm->jobqueue;
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = jobqueue->ctr;
  job->nargs          = 1;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;

  for (i=0; i<tcomm->ncommthreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
  jobqueue->kernel_ctr++;

  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

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
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  jobqueue = tcomm->jobqueue;
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = jobqueue->ctr;
  job->nargs          = 2;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;
  job->args[1]        = in2;

  for (i=0; i<tcomm->ncommthreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
  jobqueue->kernel_ctr++;

  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

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
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  jobqueue = tcomm->jobqueue;
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2,in3);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = jobqueue->ctr;
  job->nargs          = 3;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;
  job->args[1]        = in2;
  job->args[2]        = in3;

  for (i=0; i<tcomm->ncommthreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
  jobqueue->kernel_ctr++;

  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

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
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  jobqueue = tcomm->jobqueue;
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2,in3,in4);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = jobqueue->ctr;
  job->nargs          = 4;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;
  job->args[1]        = in2;
  job->args[2]        = in3;
  job->args[3]        = in4;

  for (i=0; i<tcomm->ncommthreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
  jobqueue->kernel_ctr++;

  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

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
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  jobqueue = tcomm->jobqueue;
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2,in3,in4,in5,in6);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = jobqueue->ctr;
  job->nargs          = 6;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;
  job->args[1]        = in2;
  job->args[2]        = in3;
  job->args[3]        = in4;
  job->args[4]        = in5;
  job->args[5]        = in6;


  for (i=0; i<tcomm->ncommthreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
  jobqueue->kernel_ctr++;

  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*
   Detaches the thread communicator from the MPI communicator if it exists
*/
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDetach"
PetscErrorCode PetscThreadCommDetach(MPI_Comm comm,PetscThreadComm tcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,tcomm->keyval,&ptr,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MPI_Attr_delete(comm,tcomm->keyval);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   This routine attaches the thread communicator to the MPI communicator if it does not
   exist already.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommAttach"
PetscErrorCode PetscThreadCommAttach(MPI_Comm comm,PetscThreadComm tcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,tcomm->keyval,&ptr,&flg);CHKERRQ(ierr);
  if (!flg) {
    tcomm->refct++;
    ierr = MPI_Attr_put(comm,tcomm->keyval,tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_CopyThreadComm"
/*
  This frees the thread communicator attached to MPI_Comm

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Keyval_create()
*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_CopyThreadComm(MPI_Comm comm,PetscMPIInt keyval,void *extra_state,void *attr_in,void *attr_out,int *flag)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm = (PetscThreadComm)attr_in;

  PetscFunctionBegin;
  tcomm->refct++;
  *(void**)attr_out = tcomm;

  *flag = 1;
  ierr  = PetscInfo1(0,"Copying thread communicator data in an MPI_Comm %ld\n",(long)comm);CHKERRQ(ierr);
  if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_DelThreadComm"
/*
  This frees the thread communicator attached to MPI_Comm

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Keyval_create()
*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelThreadComm(MPI_Comm comm,PetscMPIInt keyval,void *attr,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscThreadCommDestroy((PetscThreadComm*)&attr);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Deleting thread communicator data in an MPI_Comm %ld\n",(long)comm);
  if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommWorldInitialize"
/*
  PetscThreadCommWorldInitialize - Initializes the global thread communicator object

  PetscThreadCommWorldInitialize() defaults to using the nonthreaded communicator.
*/
PetscErrorCode PetscThreadCommWorldInitialize(void)
{
  PetscInt                i;
  PetscErrorCode          ierr;
  PetscThreadComm         tcomm;
  PetscThreadPool         pool;

  PetscFunctionBegin;
  ierr = PetscThreadPoolGetPool(PETSC_COMM_WORLD,&pool);
  ierr = PetscThreadCommCreate(&PETSC_THREAD_COMM_WORLD);CHKERRQ(ierr);
  tcomm = PETSC_THREAD_COMM_WORLD;

  if (pool->ismainworker) {
    tcomm->thread_start = 0;
    tcomm->ncommthreads = pool->npoolthreads;
  } else {
    tcomm->thread_start = 1;
    tcomm->ncommthreads = pool->npoolthreads-1;
  }
  ierr = PetscStrcmp(NOTHREAD,pool->type,&tcomm->isnothread);CHKERRQ(ierr);
  ierr = MPI_Keyval_create(Petsc_CopyThreadComm,Petsc_DelThreadComm,&tcomm->keyval,(void*)0);CHKERRQ(ierr);
  pool->tcworld_keyval = tcomm->keyval;
  for(i=0; i<pool->npoolthreads; i++) {
    pool->thread_keyvals[i] = pool->tcworld_keyval;
  }

  ierr = PetscMalloc1(tcomm->ncommthreads,&tcomm->my_job_counter);
  ierr = PetscMalloc1(tcomm->ncommthreads,&tcomm->my_kernel_ctr);
  ierr = PetscMalloc1(tcomm->ncommthreads,&tcomm->glob_kernel_ctr);
  for(i=0; i<tcomm->ncommthreads; i++) {
    tcomm->my_job_counter[i] = 0;
    tcomm->my_kernel_ctr[i] = 0;
    tcomm->glob_kernel_ctr[i] = 0;
  }

  /* Set the leader thread rank */
  if (pool->npoolthreads) {
    if (pool->ismainworker) tcomm->leader = pool->granks[0];
    else tcomm->leader = pool->granks[1];
  }

  ierr = PetscThreadCommCreateJobQueue(tcomm,pool);CHKERRQ(ierr);
  ierr = PetscThreadCommReductionCreate(tcomm,&tcomm->red);CHKERRQ(ierr);
  //if(pool->model==THREAD_MODEL_LOOP) {
  //ierr = PetscThreadCommStackCreate();CHKERRQ(ierr);
  //}
  tcomm->refct++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreateJobQueue"
PetscErrorCode PetscThreadCommCreateJobQueue(PetscThreadComm tcomm,PetscThreadPool pool)
{
  PetscInt i,j;
  PetscThreadCommJobQueue jobqueue;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Allocate queue
  ierr = PetscNew(&tcomm->jobqueue);
  jobqueue = tcomm->jobqueue;

  // Create job contexts
  ierr = PetscMalloc1(pool->nkernels,&jobqueue->jobs);CHKERRQ(ierr);
  ierr = PetscMalloc1(tcomm->ncommthreads*pool->nkernels,&jobqueue->jobs[0].job_status);CHKERRQ(ierr);
  for (i=0; i<pool->nkernels; i++) {
    jobqueue->jobs[i].job_status = jobqueue->jobs[0].job_status + i*tcomm->ncommthreads;
    for (j=0; j<tcomm->ncommthreads; j++) jobqueue->jobs[i].job_status[j] = THREAD_JOB_NONE;
  }

  // Set queue variables
  jobqueue->ctr = 0;
  jobqueue->kernel_ctr = 0;
  tcomm->job_ctr = 0;

  // Create thread info
  ierr = PetscMalloc1(tcomm->ncommthreads,&jobqueue->tinfo);CHKERRQ(ierr);
  for(i=0; i<tcomm->ncommthreads; i++) {
    ierr = PetscNew(&jobqueue->tinfo[i]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetOwnershipRanges"
/*
   PetscThreadCommGetOwnershipRanges - Given the global size of an array, computes the local sizes and sets
                                       the starting array indices

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
  PetscThreadComm tcomm = NULL;
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
PetscErrorCode PetscThreadCommGetRank(PetscInt *trank)
{
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscErrorCode ierr;
  PetscInt       rank = 0;

  PetscFunctionBegin;
  if (pool->ops->getrank) {
    ierr = (*pool->ops->getrank)(&rank);CHKERRQ(ierr);
  }
  *trank = rank;
  PetscFunctionReturn(0);
}
