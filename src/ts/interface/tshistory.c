#include <petsc/private/tshistoryimpl.h>

PetscClassId TSHISTORY_CLASSID;

PetscErrorCode TSHistoryGetNumSteps(TSHistory tsh, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tsh,TSHISTORY_CLASSID,1);
  PetscValidPointer(n,2);
  *n = tsh->n;
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryUpdate(TSHistory tsh, PetscInt id, PetscReal time)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tsh,TSHISTORY_CLASSID,1);
  PetscValidLogicalCollectiveInt(tsh,id,2);
  PetscValidLogicalCollectiveReal(tsh,time,3);
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

PetscErrorCode TSHistoryGetTimeStep(TSHistory tsh, PetscBool backward, PetscInt step, PetscReal *dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tsh,TSHISTORY_CLASSID,1);
  PetscValidLogicalCollectiveBool(tsh,backward,2);
  PetscValidLogicalCollectiveInt(tsh,step,3);
  PetscValidRealPointer(dt,4);
  if (step < 0 || step > tsh->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Given time step %D does not match any in history [0,%D]",step,tsh->n);
  if (!backward) *dt = tsh->hist[PetscMin(step+1,tsh->n-1)] - tsh->hist[PetscMin(step,tsh->n-1)];
  else           *dt = tsh->hist[PetscMax(tsh->n-step-1,0)] - tsh->hist[PetscMax(tsh->n-step-2,0)];
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryGetLocFromTime(TSHistory tsh, PetscReal time, PetscInt *loc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tsh,TSHISTORY_CLASSID,1);
  PetscValidLogicalCollectiveReal(tsh,time,2);
  PetscValidIntPointer(loc,3);
  if (!tsh->sorted) {
    ierr = PetscSortRealWithArrayInt(tsh->n,tsh->hist,tsh->hist_id);CHKERRQ(ierr);
    tsh->sorted = PETSC_TRUE;
  }
  ierr = PetscFindReal(time,tsh->n,tsh->hist,PETSC_SMALL,loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistorySetHistory(TSHistory tsh, PetscInt n, PetscReal hist[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tsh,TSHISTORY_CLASSID,1);
  PetscValidLogicalCollectiveInt(tsh,n,2);
  if (n) PetscValidRealPointer(hist,3);
  ierr = PetscFree(tsh->hist);CHKERRQ(ierr);
  ierr = PetscFree(tsh->hist_id);CHKERRQ(ierr);
  tsh->n = n;
  tsh->c = n;
  ierr = PetscMalloc1(tsh->n,&tsh->hist);CHKERRQ(ierr);
  ierr = PetscMalloc1(tsh->n,&tsh->hist_id);CHKERRQ(ierr);
  for (i = 0; i < tsh->n; i++) {
    tsh->hist[i]    = hist[i];
    tsh->hist_id[i] = i;
  }
  ierr = PetscSortRealWithArrayInt(tsh->n,tsh->hist,tsh->hist_id);CHKERRQ(ierr);
  tsh->sorted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryDestroy(TSHistory *tsh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*tsh) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*tsh),TSHISTORY_CLASSID,1);
  if (--((PetscObject)(*tsh))->refct > 0) {*tsh = NULL; PetscFunctionReturn(0);}
  ierr = PetscFree((*tsh)->hist);CHKERRQ(ierr);
  ierr = PetscFree((*tsh)->hist_id);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(tsh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryCreate(MPI_Comm comm, TSHistory *hst)
{
  TSHistory      tsh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(hst,2);
  *hst = NULL;
  ierr = TSInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(tsh,TSHISTORY_CLASSID,"TSHistory","Time stepping","TS",comm,TSHistoryDestroy,NULL);CHKERRQ(ierr);
  tsh->n      = 0;
  tsh->c      = 1024; /* capacity */
  tsh->s      = 1024; /* reallocation size */
  tsh->sorted = PETSC_TRUE;
  ierr = PetscMalloc1(tsh->c,&tsh->hist);CHKERRQ(ierr);
  ierr = PetscMalloc1(tsh->c,&tsh->hist_id);CHKERRQ(ierr);
  *hst = tsh;
  PetscFunctionReturn(0);
}
