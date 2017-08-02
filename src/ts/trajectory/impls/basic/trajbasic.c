
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

/* TSHistory is an helper object that allows inquiring
   the TSTrajectory by time and not by the step number only
   this can be moved to TS when consolidated, or at
   least shared by the different TSTrajectory implementations */
struct _n_TSHistory {
  PetscReal   *hist;    /* time history */
  PetscInt    *hist_id; /* stores the stepid in time history */
  PetscInt    n;        /* current number of steps stored */
  PetscBool   sorted;   /* if the history is sorted in ascending order */
  PetscReal   c;        /* current capacity of hist */
  PetscReal   s;        /* reallocation size */
};
typedef struct _n_TSHistory* TSHistory;

static PetscErrorCode TSHistoryUpdate(TSHistory tsh, PetscInt id, PetscReal time)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tsh->n == tsh->c) { /* reallocation */
    tsh->c += tsh->s;
    ierr = PetscRealloc(tsh->c*sizeof(*tsh->hist),&tsh->hist);CHKERRQ(ierr);
    ierr = PetscRealloc(tsh->c*sizeof(*tsh->hist_id),&tsh->hist_id);CHKERRQ(ierr);
  }
  tsh->sorted = (PetscBool)(tsh->sorted && (tsh->n ? time >= tsh->hist[tsh->n-1] : PETSC_TRUE));
#if defined(PETSC_USE_DEBUG)
  if (tsh->n) { /* id should be unique */
    PetscInt loc,*ids;

    ierr = PetscMalloc1(tsh->n,&ids);CHKERRQ(ierr);
    ierr = PetscMemcpy(ids,tsh->hist_id,tsh->n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscSortInt(tsh->n,ids);CHKERRQ(ierr);
    ierr = PetscFindInt(id,tsh->n,ids,&loc);CHKERRQ(ierr);
    ierr = PetscFree(ids);CHKERRQ(ierr);
    if (loc >=0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"History id should be unique");
  }
#endif
  tsh->hist[tsh->n]    = time;
  tsh->hist_id[tsh->n] = id;
  tsh->n += 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSHistoryGetIdFromTime(TSHistory tsh, PetscReal time, PetscInt *id)
{
  PetscInt       loc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tsh->sorted) {
    ierr = PetscSortRealWithArrayInt(tsh->n,tsh->hist,tsh->hist_id);CHKERRQ(ierr);
    tsh->sorted = PETSC_TRUE;
  }
  ierr = PetscFindReal(time,tsh->n,tsh->hist,PETSC_SMALL,&loc);CHKERRQ(ierr);
  *id  = loc < 0 ? loc : tsh->hist_id[loc];
  PetscFunctionReturn(0);
}

static PetscErrorCode TSHistoryDestroy(TSHistory tsh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(tsh->hist);CHKERRQ(ierr);
  ierr = PetscFree(tsh->hist_id);CHKERRQ(ierr);
  ierr = PetscFree(tsh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* these two functions are stolen from bdf.c */
PETSC_STATIC_INLINE void LagrangeBasisVals(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar L[])
{
  PetscInt k,j;
  for (k=0; k<n; k++)
    for (L[k]=1, j=0; j<n; j++)
      if (j != k)
        L[k] *= (t - T[j])/(T[k] - T[j]);
}

PETSC_STATIC_INLINE void LagrangeBasisDers(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar dL[])
{
  PetscInt  k,j,i;
  for (k=0; k<n; k++)
    for (dL[k]=0, j=0; j<n; j++)
      if (j != k) {
        PetscReal L = 1/(T[k] - T[j]);
        for (i=0; i<n; i++)
          if (i != j && i != k)
            L *= (t - T[i])/(T[k] - T[i]);
        dL[k] += L;
      }
}

typedef struct {
  /* output */
  PetscViewer viewer;
  char        *folder;
  char        *basefilename;
  char        *ext;

  /* strategy for computing from missing time */
  PetscInt    order;  /* interpolation order. if negative, recompute */
  Vec         *W;     /* work vectors */
  PetscReal   *T,*L;  /* Lagrange */

  /* history */
  TSHistory   tsh;
} TSTrajectory_Basic;

static PetscErrorCode TSTrajectoryDestroy_Basic(TSTrajectory tj)
{
  TSTrajectory_Basic *tjbasic = tj->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDestroy(&tjbasic->viewer);CHKERRQ(ierr);
  ierr = PetscFree(tjbasic->folder);CHKERRQ(ierr);
  ierr = PetscFree(tjbasic->basefilename);CHKERRQ(ierr);
  ierr = PetscFree(tjbasic->ext);CHKERRQ(ierr);
  ierr = TSHistoryDestroy(tjbasic->tsh);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tjbasic->order+1,&tjbasic->W);CHKERRQ(ierr);
  ierr = PetscFree2(tjbasic->T,tjbasic->L);CHKERRQ(ierr);
  ierr = PetscFree(tjbasic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  TSTrajectory_Basic *tjbasic = tj->data;
  char               filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode     ierr;
  MPI_Comm           comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  if (stepnum == 0) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscRMTree(tjbasic->folder);CHKERRQ(ierr);
      ierr = PetscMkdir(tjbasic->folder);CHKERRQ(ierr);
    }
    ierr = PetscBarrier((PetscObject)ts);CHKERRQ(ierr);
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s/%s-%06d.%s",tjbasic->folder,tjbasic->basefilename,stepnum,tjbasic->ext);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(tjbasic->viewer,filename);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(tjbasic->viewer);CHKERRQ(ierr);
  ierr = VecView(X,tjbasic->viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(tjbasic->viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);

  if (stepnum) {
    Vec       *Y;
    PetscReal tprev;
    PetscInt  ns,i;

    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    for (i=0;i<ns;i++) {
      ierr = VecView(Y[i],tjbasic->viewer);CHKERRQ(ierr);
    }

    ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(tjbasic->viewer,&tprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  }

  ierr = TSHistoryUpdate(tjbasic->tsh,stepnum,time);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryBasicReconstruct_Private(TSTrajectory tj,PetscReal t,PetscInt stepnum,Vec U,Vec Udot)
{
  TSTrajectory_Basic *tjbasic = tj->data;
  TSHistory          tsh = tjbasic->tsh;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (tjbasic->order < 0) {
    SETERRQ1(PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Recompute at time %g not yet implemented",t);
  } else { /* lagrange interpolation */
    PetscInt cnt = 0,i;
    if (!tjbasic->T) {
      ierr = PetscMalloc2(tjbasic->order+1,&tjbasic->T,tjbasic->order+1,&tjbasic->L);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(U,tjbasic->order+1,&tjbasic->W);CHKERRQ(ierr);
    }
    for (i = 0; i < (tjbasic->order+1)/2; i++) {
      PetscInt s = stepnum - i - 1;
      if (s > 0) {
        PetscViewer viewer;
        char        filename[PETSC_MAX_PATH_LEN];
        PetscInt    id = tsh->hist_id[s];

        ierr = PetscSNPrintf(filename,sizeof(filename),"%s/%s-%06d.%s",tjbasic->folder,tjbasic->basefilename,id,tjbasic->ext);CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
        ierr = VecLoad(tjbasic->W[cnt],viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        tjbasic->T[cnt] = tsh->hist[s];
        cnt++;
      }
    }
    for (i = 0; i < tjbasic->order/2 + 1; i++) {
      PetscInt s = stepnum + i;
      if (s < tsh->n) {
        PetscViewer viewer;
        char        filename[PETSC_MAX_PATH_LEN];
        PetscInt    id = tsh->hist_id[s];

        ierr = PetscSNPrintf(filename,sizeof(filename),"%s/%s-%06d.%s",tjbasic->folder,tjbasic->basefilename,id,tjbasic->ext);CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
        ierr = VecLoad(tjbasic->W[cnt],viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        tjbasic->T[cnt] = tsh->hist[s];
        cnt++;
      }
    }
    if (U) {
      LagrangeBasisVals(cnt,t,tjbasic->T,tjbasic->L);
      ierr = VecZeroEntries(U);CHKERRQ(ierr);
      ierr = VecMAXPY(U,cnt,tjbasic->L,tjbasic->W);CHKERRQ(ierr);
    }
    if (Udot) {
      LagrangeBasisDers(cnt,t,tjbasic->T,tjbasic->L);
      ierr = VecZeroEntries(Udot);CHKERRQ(ierr);
      ierr = VecMAXPY(Udot,cnt,tjbasic->L,tjbasic->W);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGetVecs_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t,Vec U,Vec Udot)
{
  TSTrajectory_Basic *tjbasic = tj->data;
  PetscErrorCode     ierr;
  PetscViewer        viewer;
  char               filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (stepnum < 0 || Udot) { /* reverse search for requested time in TSHistory*/
    TSHistory tsh = tjbasic->tsh;

    ierr = TSHistoryGetIdFromTime(tsh,*t,&stepnum);CHKERRQ(ierr);
    if (stepnum == -1 || stepnum == -tsh->n - 1) {
      PetscReal t0 = tsh->n ? tsh->hist[0]        : 0.0;
      PetscReal tf = tsh->n ? tsh->hist[tsh->n-1] : 0.0;
      SETERRQ5(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Requested time %g (%d) is outside the history interval [%g, %g] (%d)",*t,stepnum,t0,tf,tsh->n);
    }
    stepnum = stepnum < 0 ? -(stepnum+1) : stepnum;
    ierr = TSTrajectoryBasicReconstruct_Private(tj,*t,stepnum,U,Udot);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* we can safely load from stepnum */
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s/%s-%06d.%s",tjbasic->folder,tjbasic->basefilename,stepnum,tjbasic->ext);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(U,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,t,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  TSTrajectory_Basic *tjbasic = tj->data;
  PetscViewer        viewer;
  char               filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode     ierr;
  Vec                Sol;

  PetscFunctionBegin;
  ierr = TSGetSolution(ts,&Sol);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s/%s-%06d.%s",tjbasic->folder,tjbasic->basefilename,stepnum,tjbasic->ext);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(Sol,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,t,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  if (stepnum != 0) {
    Vec         *Y;
    PetscInt    Nr,i;
    PetscReal   timepre;

    ierr = TSGetStages(ts,&Nr,&Y);CHKERRQ(ierr);
    for (i=0;i<Nr ;i++) {
      ierr = VecLoad(Y[i],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryRead(viewer,&timepre,1,NULL,PETSC_REAL);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-(*t)+timepre);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYBASIC - Stores each solution of the ODE/DAE in a file

      Saves each timestep into a seperate file in SA-data/SA-%06d.bin

      This version saves the solutions at all the stages

      $PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m can read in files created with this format

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory tj,TS ts)
{
  TSTrajectory_Basic *tjbasic;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&tjbasic);CHKERRQ(ierr);

  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)ts),&tjbasic->viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(tjbasic->viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(tjbasic->viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscStrallocpy("./SA-data",&tjbasic->folder);CHKERRQ(ierr);
  ierr = PetscStrallocpy("SA",&tjbasic->basefilename);CHKERRQ(ierr);
  ierr = PetscStrallocpy("bin",&tjbasic->ext);CHKERRQ(ierr);

  ierr = PetscNew(&tjbasic->tsh);CHKERRQ(ierr);
  tjbasic->tsh->n      = 0;
  tjbasic->tsh->c      = 1000;
  tjbasic->tsh->s      = 1000;
  tjbasic->tsh->sorted = PETSC_TRUE;
  ierr = PetscMalloc1(tjbasic->tsh->c,&tjbasic->tsh->hist);CHKERRQ(ierr);
  ierr = PetscMalloc1(tjbasic->tsh->c,&tjbasic->tsh->hist_id);CHKERRQ(ierr);

  tjbasic->order = 1;

  tj->data = tjbasic;

  tj->ops->set     = TSTrajectorySet_Basic;
  tj->ops->get     = TSTrajectoryGet_Basic;
  tj->ops->getvecs = TSTrajectoryGetVecs_Basic;
  tj->ops->destroy = TSTrajectoryDestroy_Basic;
  PetscFunctionReturn(0);
}
