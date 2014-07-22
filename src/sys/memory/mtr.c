
/*
     Interface to malloc() and free(). This code allows for
  logging of memory usage and some error checking
*/
#include <petscsys.h>           /*I "petscsys.h" I*/
#include <petscviewer.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

/*
     These are defined in mal.c and ensure that malloced space is PetscScalar aligned
*/
extern PetscErrorCode  PetscMallocAlign(size_t,int,const char[],const char[],void**);
extern PetscErrorCode  PetscFreeAlign(void*,int,const char[],const char[]);
extern PetscErrorCode  PetscTrMallocDefault(size_t,int,const char[],const char[],void**);
extern PetscErrorCode  PetscTrFreeDefault(void*,int,const char[],const char[]);

#define CLASSID_VALUE   ((PetscClassId) 0xf0e0d0c9)
#define ALREADY_FREED  ((PetscClassId) 0x0f0e0d9c)

/* HEADER_BYTES is the number of bytes in a PetscMalloc() header.
   It is sizeof(TRSPACE) padded to be a multiple of PETSC_MEMALIGN.
*/

#define HEADER_BYTES      ((sizeof(TRSPACE)+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1))

typedef struct _trSPACE {
  size_t       size;
  int          id;
  int          lineno;
  const char   *filename;
  const char   *functionname;
  PetscClassId classid;
#if defined(PETSC_USE_DEBUG)
  PetscStack   stack;
#endif
  struct _trSPACE *next,*prev;
} TRSPACE;

/* This union is used to insure that the block passed to the user retains
   a minimum alignment of PETSC_MEMALIGN.
*/
typedef union {
  TRSPACE sp;
  char    v[HEADER_BYTES];
} TrSPACE;

typedef struct {
  size_t    TRallocated;
  int       TRfrags;
  TRSPACE   *TRhead;
  int       TRid;
  PetscBool TRdebugLevel;
  size_t    TRMaxMem;

  /* Arrays to log information on all Mallocs */
  int        PetscLogMallocMax,PetscLogMalloc;
  size_t     PetscLogMallocThreshold;
  size_t     *PetscLogMallocLength;
  const char **PetscLogMallocFile,**PetscLogMallocFunction;
} PetscTRMallocStruct;

#if defined(PETSC_HAVE_PTHREADCLASSES)
#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscTRMallocStruct *trmalloc;
#else
PetscThreadKey trmalloc;
#endif
#elif defined(PETSC_HAVE_OPENMP)
PetscTRMallocStruct *trmalloc;
#pragma omp threadprivate(trmalloc)
#else
PetscTRMallocStruct *trmalloc;
#endif

/* Global malloc statistics */
int global_TRallocated = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscSetUseTrMalloc_Private"
PetscErrorCode PetscSetUseTrMalloc_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Return if trmalloc has already been created */
  if(trmalloc) PetscFunctionReturn(0);

  /* Allocate and initialize trmalloc */
  ierr = PetscNew(&trmalloc);CHKERRQ(ierr);
  ierr = PetscMallocSet(PetscTrMallocDefault,PetscTrFreeDefault);CHKERRQ(ierr);

  trmalloc->TRallocated             = 0;
  trmalloc->TRfrags                 = 0;
  trmalloc->TRhead                  = 0;
  trmalloc->TRid                    = 0;
  trmalloc->TRdebugLevel            = PETSC_FALSE;
  trmalloc->TRMaxMem                = 0;
  trmalloc->PetscLogMallocMax       = 10000;
  trmalloc->PetscLogMalloc          = -1;
  trmalloc->PetscLogMallocThreshold = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMallocValidate"
/*@C
   PetscMallocValidate - Test the memory for corruption.  This can be used to
   check for memory overwrites.

   Input Parameter:
+  line - line number where call originated.
.  function - name of function calling
-  file - file where function is

   Return value:
   The number of errors detected.

   Output Effect:
   Error messages are written to stdout.

   Level: advanced

   Notes:
    You should generally use CHKMEMQ as a short cut for calling this
    routine.

    The line, function, file are given by the C preprocessor as
    __LINE__, __FUNCT__, __FILE__

    The Fortran calling sequence is simply PetscMallocValidate(ierr)

   No output is generated if there are no problems detected.

.seealso: CHKMEMQ

@*/
PetscErrorCode  PetscMallocValidate(int line,const char function[],const char file[])
{
  TRSPACE      *head,*lasthead;
  char         *a;
  PetscClassId *nend;

  PetscFunctionBegin;
  head = trmalloc->TRhead; lasthead = NULL;
  while (head) {
    if (head->classid != CLASSID_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at  %s() line %d in %s\n",function,line,file);
      (*PetscErrorPrintf)("Memory at address %p is corrupted\n",head);
      (*PetscErrorPrintf)("Probably write past beginning or end of array\n");
      if (lasthead) (*PetscErrorPrintf)("Last intact block allocated in %s() line %d in %s\n",lasthead->functionname,lasthead->lineno,lasthead->filename);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC," ");
    }
    a    = (char*)(((TrSPACE*)head) + 1);
    nend = (PetscClassId*)(a + head->size);
    if (*nend != CLASSID_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at %s() line %d in %s\n",function,line,file);
      if (*nend == ALREADY_FREED) {
        (*PetscErrorPrintf)("Memory [id=%d(%.0f)] at address %p already freed\n",head->id,(PetscLogDouble)head->size,a);
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC," ");
      } else {
        (*PetscErrorPrintf)("Memory [id=%d(%.0f)] at address %p is corrupted (probably write past end of array)\n",head->id,(PetscLogDouble)head->size,a);
        (*PetscErrorPrintf)("Memory originally allocated in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC," ");
      }
    }
    lasthead = head;
    head     = head->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscTrMallocDefault"
/*
    PetscTrMallocDefault - Malloc with tracing.

    Input Parameters:
+   a   - number of bytes to allocate
.   lineno - line number where used.  Use __LINE__ for this
.   function - function calling routine. Use __FUNCT__ for this
-   filename  - file name where used.  Use __FILE__ for this

    Returns:
    double aligned pointer to requested storage, or null if not
    available.
 */
PetscErrorCode  PetscTrMallocDefault(size_t a,int lineno,const char function[],const char filename[],void **result)
{
  TRSPACE        *head;
  char           *inew;
  size_t         nsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (trmalloc->TRdebugLevel) {
    ierr = PetscMallocValidate(lineno,function,filename); if (ierr) PetscFunctionReturn(ierr);
  }

  nsize = (a + (PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1);
  ierr  = PetscMallocAlign(nsize+sizeof(TrSPACE)+sizeof(PetscClassId),lineno,function,filename,(void**)&inew);CHKERRQ(ierr);

  head  = (TRSPACE*)inew;
  inew += sizeof(TrSPACE);

  if (trmalloc->TRhead) trmalloc->TRhead->prev = head;
  head->next       = trmalloc->TRhead;
  trmalloc->TRhead = head;
  head->prev       = 0;
  head->size       = nsize;
  head->id         = trmalloc->TRid;
  head->lineno     = lineno;

  head->filename                 = filename;
  head->functionname             = function;
  head->classid                  = CLASSID_VALUE;
  *(PetscClassId*)(inew + nsize) = CLASSID_VALUE;

  trmalloc->TRallocated += nsize;
  if (trmalloc->TRallocated > trmalloc->TRMaxMem) {
    trmalloc->TRMaxMem = trmalloc->TRallocated;
  }
  trmalloc->TRfrags++;

#if defined(PETSC_USE_DEBUG)
  if (PetscStackActive()) {
    ierr = PetscStackCopy((PetscStack*)PetscThreadLocalGetValue(petscstack),&head->stack);CHKERRQ(ierr);
    /* fix the line number to where the malloc() was called, not the PetscFunctionBegin; */
    head->stack.line[head->stack.currentsize-2] = lineno;
  }
#endif

  /*
         Allow logging of all mallocs made
  */
  if (trmalloc->PetscLogMalloc > -1 && trmalloc->PetscLogMalloc < trmalloc->PetscLogMallocMax && a >= trmalloc->PetscLogMallocThreshold) {
    if (!trmalloc->PetscLogMalloc) {
      trmalloc->PetscLogMallocLength = (size_t*)malloc(trmalloc->PetscLogMallocMax*sizeof(size_t));
      if (!trmalloc->PetscLogMallocLength) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM," ");

      trmalloc->PetscLogMallocFile = (const char**)malloc(trmalloc->PetscLogMallocMax*sizeof(char*));
      if (!trmalloc->PetscLogMallocFile) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM," ");

      trmalloc->PetscLogMallocFunction = (const char**)malloc(trmalloc->PetscLogMallocMax*sizeof(char*));
      if (!trmalloc->PetscLogMallocFunction) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM," ");
    }
    trmalloc->PetscLogMallocLength[trmalloc->PetscLogMalloc]     = nsize;
    trmalloc->PetscLogMallocFile[trmalloc->PetscLogMalloc]       = filename;
    trmalloc->PetscLogMallocFunction[trmalloc->PetscLogMalloc++] = function;
  }
  *result = (void*)inew;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscTrFreeDefault"
/*
   PetscTrFreeDefault - Free with tracing.

   Input Parameters:
.   a    - pointer to a block allocated with PetscTrMalloc
.   lineno - line number where used.  Use __LINE__ for this
.   function - function calling routine. Use __FUNCT__ for this
.   file  - file name where used.  Use __FILE__ for this
 */
PetscErrorCode  PetscTrFreeDefault(void *aa,int line,const char function[],const char file[])
{
  char           *a = (char*)aa;
  TRSPACE        *head;
  char           *ahead;
  PetscErrorCode ierr;
  PetscClassId   *nend;

  PetscFunctionBegin;
  /* Do not try to handle empty blocks */
  if (!a) PetscFunctionReturn(0);

  if (trmalloc->TRdebugLevel) {
    ierr = PetscMallocValidate(line,function,file);CHKERRQ(ierr);
  }

  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE*)a;

  if (head->classid != CLASSID_VALUE) {
    (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s\n",function,line,file);
    (*PetscErrorPrintf)("Block at address %p is corrupted; cannot free;\nmay be block not allocated with PetscMalloc()\n",a);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Bad location or corrupted memory");
  }
  nend = (PetscClassId*)(ahead + head->size);
  if (*nend != CLASSID_VALUE) {
    if (*nend == ALREADY_FREED) {
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s\n",function,line,file);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p was already freed\n",head->id,(PetscLogDouble)head->size,a + sizeof(TrSPACE));
      if (head->lineno > 0 && head->lineno < 50000 /* sanity check */) {
        (*PetscErrorPrintf)("Block freed in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
      } else {
        (*PetscErrorPrintf)("Block allocated in %s() line %d in %s\n",head->functionname,-head->lineno,head->filename);
      }
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Memory already freed");
    } else {
      /* Damaged tail */
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s\n",function,line,file);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p is corrupted (probably write past end of array)\n",head->id,(PetscLogDouble)head->size,a);
      (*PetscErrorPrintf)("Block allocated in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Corrupted memory");
    }
  }
  /* Mark the location freed */
  *nend = ALREADY_FREED;
  /* Save location where freed.  If we suspect the line number, mark as  allocated location */
  if (line > 0 && line < 50000) {
    head->lineno       = line;
    head->filename     = file;
    head->functionname = function;
  } else {
    head->lineno = -head->lineno;
  }
  /* zero out memory - helps to find some reuse of already freed memory */
  ierr = PetscMemzero(aa,head->size);CHKERRQ(ierr);

  trmalloc->TRallocated -= head->size;
  trmalloc->TRfrags--;
  if (head->prev) head->prev->next = head->next;
  else trmalloc->TRhead = head->next;

  if (head->next) head->next->prev = head->prev;
  ierr = PetscFreeAlign(a,line,function,file);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscMemoryShowUsage"
/*@C
    PetscMemoryShowUsage - Shows the amount of memory currently being used
        in a communicator.

    Collective on PetscViewer

    Input Parameter:
+    viewer - the viewer that defines the communicator
-    message - string printed before values

    Level: intermediate

    Concepts: memory usage

.seealso: PetscMallocDump(), PetscMemoryGetCurrentUsage()
 @*/
PetscErrorCode  PetscMemoryShowUsage(PetscViewer viewer,const char message[])
{
  PetscLogDouble allocated,maximum,resident,residentmax;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr = PetscMallocGetCurrentUsage(&allocated);CHKERRQ(ierr);
  ierr = PetscMallocGetMaximumUsage(&maximum);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&resident);CHKERRQ(ierr);
  ierr = PetscMemoryGetMaximumUsage(&residentmax);CHKERRQ(ierr);
  if (residentmax > 0) residentmax = PetscMax(resident,residentmax);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,message);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
  if (resident && residentmax && allocated) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Current space PetscMalloc()ed %g, max space PetscMalloced() %g\n[%d]Current process memory %g max process memory %g\n",rank,allocated,maximum,rank,resident,residentmax);CHKERRQ(ierr);
  } else if (resident && residentmax) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Run with -malloc to get statistics on PetscMalloc() calls\n[%d]Current process memory %g max process memory %g\n",rank,rank,resident,residentmax);CHKERRQ(ierr);
  } else if (resident && allocated) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Current space PetscMalloc()ed %g, max space PetscMalloced() %g\n[%d]Current process memory %g, run with -memory_info to get max memory usage\n",rank,allocated,maximum,rank,resident);CHKERRQ(ierr);
  } else if (allocated) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Current space PetscMalloc()ed %g, max space PetscMalloced() %g\n[%d]OS cannot compute process memory\n",rank,allocated,maximum,rank);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"Run with -malloc to get statistics on PetscMalloc() calls\nOS cannot compute process memory\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMallocGetCurrentUsage"
/*@C
    PetscMallocGetCurrentUsage - gets the current amount of memory used that was PetscMalloc()ed

    Not Collective

    Output Parameters:
.   space - number of bytes currently allocated

    Level: intermediate

    Concepts: memory usage

.seealso: PetscMallocDump(), PetscMallocDumpLog(), PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(),
          PetscMemoryGetMaximumUsage()
 @*/
PetscErrorCode  PetscMallocGetCurrentUsage(PetscLogDouble *space)
{
  PetscFunctionBegin;
  *space = (PetscLogDouble) trmalloc->TRallocated;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMallocGetMaximumUsage"
/*@C
    PetscMallocGetMaximumUsage - gets the maximum amount of memory used that was PetscMalloc()ed at any time
        during this run.

    Not Collective

    Output Parameters:
.   space - maximum number of bytes ever allocated at one time

    Level: intermediate

    Concepts: memory usage

.seealso: PetscMallocDump(), PetscMallocDumpLog(), PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(),
          PetscMemoryGetCurrentUsage()
 @*/
PetscErrorCode  PetscMallocGetMaximumUsage(PetscLogDouble *space)
{
  PetscFunctionBegin;
  *space = (PetscLogDouble) trmalloc->TRMaxMem;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
#undef __FUNCT__
#define __FUNCT__ "PetscMallocGetStack"
/*@C
   PetscMallocGetStack - returns a pointer to the stack for the location in the program a call to PetscMalloc() was used to obtain that memory

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.    ptr - the memory location

   Output Paramter:
.    stack - the stack indicating where the program allocated this memory

   Level: intermediate

.seealso:  PetscMallocGetCurrentUsage(), PetscMallocDumpLog()
@*/
PetscErrorCode  PetscMallocGetStack(void *ptr,PetscStack **stack)
{
  TRSPACE *head;

  PetscFunctionBegin;
  head   = (TRSPACE*) (((char*)ptr) - HEADER_BYTES);
  *stack = &head->stack;
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "PetscMallocGetStack"
PetscErrorCode  PetscMallocGetStack(void *ptr,void **stack)
{
  PetscFunctionBegin;
  *stack = 0;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscMallocDump"
/*@C
   PetscMallocDump - Dumps the allocated memory blocks to a file. The information
   printed is: size of space (in bytes), address of space, id of space,
   file in which space was allocated, and line number at which it was
   allocated.

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  fp  - file pointer.  If fp is NULL, stdout is assumed.

   Options Database Key:
.  -malloc_dump - Dumps unfreed memory during call to PetscFinalize()

   Level: intermediate

   Fortran Note:
   The calling sequence in Fortran is PetscMallocDump(integer ierr)
   The fp defaults to stdout.

   Notes: uses MPI_COMM_WORLD, because this may be called in PetscFinalize() after PETSC_COMM_WORLD
          has been freed.

   Concepts: memory usage
   Concepts: memory bleeding
   Concepts: bleeding memory

.seealso:  PetscMallocGetCurrentUsage(), PetscMallocDumpLog()
@*/
PetscErrorCode  PetscMallocDump(FILE *fp)
{
  TRSPACE        *head;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!fp) fp = PETSC_STDOUT;
  if (trmalloc->TRallocated > 0) fprintf(fp,"[%d]Total space allocated %.0f bytes\n",rank,(PetscLogDouble)trmalloc->TRallocated);
  head = trmalloc->TRhead;
  while (head) {
    fprintf(fp,"[%2d]%.0f bytes %s() line %d in %s\n",rank,(PetscLogDouble)head->size,head->functionname,head->lineno,head->filename);
#if defined(PETSC_USE_DEBUG)
    ierr = PetscStackPrint(&head->stack,fp);CHKERRQ(ierr);
#endif
    head = head->next;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PetscMallocSetDumpLog"
/*@C
    PetscMallocSetDumpLog - Activates logging of all calls to PetscMalloc().

    Not Collective

    Options Database Key:
+  -malloc_log <filename> - Activates PetscMallocDumpLog()
-  -malloc_log_threshold <min> - Activates logging and sets a minimum size

    Level: advanced

.seealso: PetscMallocDump(), PetscMallocDumpLog(), PetscMallocSetDumpLogThreshold()
@*/
PetscErrorCode PetscMallocSetDumpLog(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  trmalloc->PetscLogMalloc = 0;

  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMallocSetDumpLogThreshold"
/*@C
    PetscMallocSetDumpLogThreshold - Activates logging of all calls to PetscMalloc().

    Not Collective

    Input Arguments:
.   logmin - minimum allocation size to log, or PETSC_DEFAULT

    Options Database Key:
+  -malloc_log <filename> - Activates PetscMallocDumpLog()
-  -malloc_log_threshold <min> - Activates logging and sets a minimum size

    Level: advanced

.seealso: PetscMallocDump(), PetscMallocDumpLog(), PetscMallocSetDumpLog()
@*/
PetscErrorCode PetscMallocSetDumpLogThreshold(PetscLogDouble logmin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMallocSetDumpLog();CHKERRQ(ierr);
  if (logmin < 0) logmin = 0.0; /* PETSC_DEFAULT or PETSC_DECIDE */
  trmalloc->PetscLogMallocThreshold = (size_t)logmin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMallocGetDumpLog"
/*@C
    PetscMallocGetDumpLog - Determine whether all calls to PetscMalloc() are being logged

    Not Collective

    Output Arguments
.   logging - PETSC_TRUE if logging is active

    Options Database Key:
.  -malloc_log - Activates PetscMallocDumpLog()

    Level: advanced

.seealso: PetscMallocDump(), PetscMallocDumpLog()
@*/
PetscErrorCode PetscMallocGetDumpLog(PetscBool *logging)
{

  PetscFunctionBegin;
  *logging = (PetscBool)(trmalloc->PetscLogMalloc >= 0);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMallocDumpLog"
/*@C
    PetscMallocDumpLog - Dumps the log of all calls to PetscMalloc(); also calls
       PetscMemoryGetMaximumUsage()

    Collective on PETSC_COMM_WORLD

    Input Parameter:
.   fp - file pointer; or NULL

    Options Database Key:
.  -malloc_log - Activates PetscMallocDumpLog()

    Level: advanced

   Fortran Note:
   The calling sequence in Fortran is PetscMallocDumpLog(integer ierr)
   The fp defaults to stdout.

.seealso: PetscMallocGetCurrentUsage(), PetscMallocDump(), PetscMallocSetDumpLog()
@*/
PetscErrorCode  PetscMallocDumpLog(FILE *fp)
{
  PetscInt       i,j,n,dummy,*perm;
  size_t         *shortlength;
  int            *shortcount,err;
  PetscMPIInt    rank,size,tag = 1212 /* very bad programming */;
  PetscBool      match;
  const char     **shortfunction;
  PetscLogDouble rss;
  MPI_Status     status;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRQ(ierr);
  /*
       Try to get the data printed in order by processor. This will only sometimes work
  */
  err = fflush(fp);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");

  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  if (rank) {
    ierr = MPI_Recv(&dummy,1,MPIU_INT,rank-1,tag,MPI_COMM_WORLD,&status);CHKERRQ(ierr);
  }

  if (trmalloc->PetscLogMalloc < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscMallocDumpLog() called without call to PetscMallocSetDumpLog() this is often due to\n                      setting the option -malloc_log AFTER PetscInitialize() with PetscOptionsInsert() or PetscOptionsInsertFile()");

  if (!fp) fp = PETSC_STDOUT;
  ierr = PetscMemoryGetMaximumUsage(&rss);CHKERRQ(ierr);
  if (rss) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Maximum memory PetscMalloc()ed %.0f maximum size of entire process %.0f\n",rank,(PetscLogDouble)trmalloc->TRMaxMem,rss);CHKERRQ(ierr);
  } else {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Maximum memory PetscMalloc()ed %.0f OS cannot compute size of entire process\n",rank,(PetscLogDouble)trmalloc->TRMaxMem);CHKERRQ(ierr);
  }
  shortcount    = (int*)malloc(trmalloc->PetscLogMalloc*sizeof(int));if (!shortcount) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Out of memory");
  shortlength   = (size_t*)malloc(trmalloc->PetscLogMalloc*sizeof(size_t));if (!shortlength) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Out of memory");
  shortfunction = (const char**)malloc(trmalloc->PetscLogMalloc*sizeof(char*));if (!shortfunction) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Out of memory");
  for (i=0,n=0; i<trmalloc->PetscLogMalloc; i++) {
    for (j=0; j<n; j++) {
      ierr = PetscStrcmp(shortfunction[j],trmalloc->PetscLogMallocFunction[i],&match);CHKERRQ(ierr);
      if (match) {
        shortlength[j] += trmalloc->PetscLogMallocLength[i];
        shortcount[j]++;
        goto foundit;
      }
    }
    shortfunction[n] = trmalloc->PetscLogMallocFunction[i];
    shortlength[n]   = trmalloc->PetscLogMallocLength[i];
    shortcount[n]    = 1;
    n++;
foundit:;
  }

  perm = (PetscInt*)malloc(n*sizeof(PetscInt));if (!perm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Out of memory");
  for (i=0; i<n; i++) perm[i] = i;
  ierr = PetscSortStrWithPermutation(n,(const char**)shortfunction,perm);CHKERRQ(ierr);

  ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Memory usage sorted by function\n",rank);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] %d %.0f %s()\n",rank,shortcount[perm[i]],(PetscLogDouble)shortlength[perm[i]],shortfunction[perm[i]]);CHKERRQ(ierr);
  }
  free(perm);
  free(shortlength);
  free(shortcount);
  free((char**)shortfunction);
  err = fflush(fp);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  if (rank != size-1) {
    ierr = MPI_Send(&dummy,1,MPIU_INT,rank+1,tag,MPI_COMM_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PetscMallocDebug"
/*@C
    PetscMallocDebug - Turns on/off debugging for the memory management routines.

    Not Collective

    Input Parameter:
.   level - PETSC_TRUE or PETSC_FALSE

   Level: intermediate

.seealso: CHKMEMQ(), PetscMallocValidate()
@*/
PetscErrorCode  PetscMallocDebug(PetscBool level)
{
  PetscFunctionBegin;
  trmalloc->TRdebugLevel = level;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMallocGetDebug"
/*@C
    PetscMallocGetDebug - Indicates if any PETSc is doing ANY memory debugging.

    Not Collective

    Output Parameter:
.    flg - PETSC_TRUE if any debugger

   Level: intermediate

    Note that by default, the debug version always does some debugging unless you run with -malloc no


.seealso: CHKMEMQ(), PetscMallocValidate()
@*/
PetscErrorCode  PetscMallocGetDebug(PetscBool *flg)
{
  PetscFunctionBegin;
  if (PetscTrMalloc == PetscTrMallocDefault) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscTrMallocMergeData"
PetscErrorCode PetscTrMallocMergeData(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscThreadLockAcquire(PetscLocks->trmalloc_lock);CHKERRQ(ierr);
  global_TRallocated += trmalloc->TRallocated;
  ierr = PetscThreadLockRelease(PetscLocks->trmalloc_lock);CHKERRQ(ierr);
  trmalloc->TRallocated = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscTrMallocDestroy"
PetscErrorCode PetscTrMallocDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTrMallocMergeData();CHKERRQ(ierr);
  ierr = PetscFree(trmalloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
