
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

typedef struct {
  PetscViewer viewer;
  char        *folder;
  char        *basefilename;
  char        *ext;
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
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  TSTrajectory_Basic *tjbasic = tj->data;
  PetscViewer        viewer;
  Vec                Sol;
  char               filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s/%s-%06d.%s",tjbasic->folder,tjbasic->basefilename,stepnum,tjbasic->ext);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  ierr = TSGetSolution(ts,&Sol);CHKERRQ(ierr);
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
  tj->data = tjbasic;

  tj->ops->set     = TSTrajectorySet_Basic;
  tj->ops->get     = TSTrajectoryGet_Basic;
  tj->ops->destroy = TSTrajectoryDestroy_Basic;
  PetscFunctionReturn(0);
}
