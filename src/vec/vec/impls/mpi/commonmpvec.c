
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "VecGhostStateSync_Private"
/*
  This is used in VecGhostGetLocalForm and VecGhostRestoreLocalForm to ensure
  that the state is updated if either vector has changed since the last time
  one of these functions was called.  It could apply to any PetscObject, but
  VecGhost is quite different from other objects in that two separate vectors
  look at the same memory.

  In principle, we could only propagate state to the local vector on
  GetLocalForm and to the global vector on RestoreLocalForm, but this version is
  more conservative (i.e. robust against misuse) and simpler.

  Note that this function is correct and changes nothing if both arguments are the
  same, which is the case in serial.
*/
static PetscErrorCode VecGhostStateSync_Private(Vec g,Vec l)
{
  PetscErrorCode   ierr;
  PetscObjectState gstate,lstate;

  PetscFunctionBegin;
  ierr = PetscObjectStateGet((PetscObject)g,&gstate);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)l,&lstate);CHKERRQ(ierr);
  ierr = PetscObjectStateSet((PetscObject)g,PetscMax(gstate,lstate));CHKERRQ(ierr);
  ierr = PetscObjectStateSet((PetscObject)l,PetscMax(gstate,lstate));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecGhostGetLocalForm"
/*@
    VecGhostGetLocalForm - Obtains the local ghosted representation of
    a parallel vector (obtained with VecCreateGhost(), VecCreateGhostWithArray()
    or VecCreateSeq()). Returns NULL if the Vec is not ghosted.

    Logically Collective

    Input Parameter:
.   g - the global vector

    Output Parameter:
.   l - the local (ghosted) representation, NULL if g is not ghosted

    Notes:
    This routine does not actually update the ghost values, but rather it
    returns a sequential vector that includes the locations for the ghost
    values and their current values. The returned vector and the original
    vector passed in share the same array that contains the actual vector data.

    To update the ghost values from the locations on the other processes one must call
    VecGhostUpdateBegin() and VecGhostUpdateEnd() before accessing the ghost values. Thus normal
    usage is
$     VecGhostUpdateBegin(x,INSERT_VALUES,SCATTER_FORWARD);
$     VecGhostUpdateEnd(x,INSERT_VALUES,SCATTER_FORWARD);
$     VecGhostGetLocalForm(x,&xlocal);
$     VecGetArray(xlocal,&xvalues);
$        // access the non-ghost values in locations xvalues[0:n-1] and ghost values in locations xvalues[n:n+nghost];
$     VecRestoreArray(xlocal,&xvalues);
$     VecGhostRestoreLocalForm(x,&xlocal);

    One should call VecGhostRestoreLocalForm() or VecDestroy() once one is
    finished using the object.

    Level: advanced

   Concepts: vectors^ghost point access

.seealso: VecCreateGhost(), VecGhostRestoreLocalForm(), VecCreateGhostWithArray()

@*/
PetscErrorCode  VecGhostGetLocalForm(Vec g,Vec *l)
{
  PetscErrorCode ierr;
  PetscBool      isseq,ismpi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_CLASSID,1);
  PetscValidPointer(l,2);

  ierr = PetscObjectTypeCompare((PetscObject)g,VECSEQ,&isseq);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)g,VECMPI,&ismpi);CHKERRQ(ierr);
  if (ismpi) {
    Vec_MPI *v = (Vec_MPI*)g->data;
    *l = v->localrep;
  } else if (isseq) {
    *l = g;
  } else {
    *l = NULL;
  }
  if (*l) {
    ierr = VecGhostStateSync_Private(g,*l);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGhostIsLocalForm"
/*@
    VecGhostIsLocalForm - Checks if a given vector is the local form of a global vector

    Not Collective

    Input Parameter:
+   g - the global vector
-   l - the local vector

    Output Parameter:
.   flg - PETSC_TRUE if local vector is local form

    Level: advanced

   Concepts: vectors^ghost point access

.seealso: VecCreateGhost(), VecGhostRestoreLocalForm(), VecCreateGhostWithArray(), VecGhostGetLocalForm()

@*/
PetscErrorCode VecGhostIsLocalForm(Vec g,Vec l,PetscBool *flg)
{
  PetscErrorCode ierr;
  PetscBool      isseq,ismpi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,1);

  *flg = PETSC_FALSE;
  ierr = PetscObjectTypeCompare((PetscObject)g,VECSEQ,&isseq);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)g,VECMPI,&ismpi);CHKERRQ(ierr);
  if (ismpi) {
    Vec_MPI *v = (Vec_MPI*)g->data;
    if (l == v->localrep) *flg = PETSC_TRUE;
  } else if (isseq) {
    if (l == g) *flg = PETSC_TRUE;
  } else SETERRQ(PetscObjectComm((PetscObject)g),PETSC_ERR_ARG_WRONG,"Global vector is not ghosted");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGhostRestoreLocalForm"
/*@
    VecGhostRestoreLocalForm - Restores the local ghosted representation of
    a parallel vector obtained with VecGhostGetLocalForm().

    Logically Collective

    Input Parameter:
+   g - the global vector
-   l - the local (ghosted) representation

    Notes:
    This routine does not actually update the ghost values, but rather it
    returns a sequential vector that includes the locations for the ghost values
    and their current values.

    Level: advanced

.seealso: VecCreateGhost(), VecGhostGetLocalForm(), VecCreateGhostWithArray()
@*/
PetscErrorCode  VecGhostRestoreLocalForm(Vec g,Vec *l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (*l) {
    ierr = VecGhostStateSync_Private(g,*l);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)*l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGhostUpdateBegin"
/*@
   VecGhostUpdateBegin - Begins the vector scatter to update the vector from
   local representation to global or global representation to local.

   Neighbor-wise Collective on Vec

   Input Parameters:
+  g - the vector (obtained with VecCreateGhost() or VecDuplicate())
.  insertmode - one of ADD_VALUES or INSERT_VALUES
-  scattermode - one of SCATTER_FORWARD or SCATTER_REVERSE

   Notes:
   Use the following to update the ghost regions with correct values from the owning process
.vb
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Use the following to accumulate the ghost region values onto the owning processors
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
.ve

   To accumulate the ghost region values onto the owning processors and then update
   the ghost regions correctly, call the later followed by the former, i.e.,
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Level: advanced

.seealso: VecCreateGhost(), VecGhostUpdateEnd(), VecGhostGetLocalForm(),
          VecGhostRestoreLocalForm(),VecCreateGhostWithArray()

@*/
PetscErrorCode  VecGhostUpdateBegin(Vec g,InsertMode insertmode,ScatterMode scattermode)
{
  Vec_MPI        *v;
  PetscErrorCode ierr;
  PetscBool      ismpi,isseq;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)g,VECMPI,&ismpi);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)g,VECSEQ,&isseq);CHKERRQ(ierr);
  if (ismpi) {
    v = (Vec_MPI*)g->data;
    if (!v->localrep) SETERRQ(PetscObjectComm((PetscObject)g),PETSC_ERR_ARG_WRONG,"Vector is not ghosted");
    if (!v->localupdate) PetscFunctionReturn(0);
    if (scattermode == SCATTER_REVERSE) {
      ierr = VecScatterBegin(v->localupdate,v->localrep,g,insertmode,scattermode);CHKERRQ(ierr);
    } else {
      ierr = VecScatterBegin(v->localupdate,g,v->localrep,insertmode,scattermode);CHKERRQ(ierr);
    }
  } else if (isseq) {
    /* Do nothing */
  } else SETERRQ(PetscObjectComm((PetscObject)g),PETSC_ERR_ARG_WRONG,"Vector is not ghosted");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGhostUpdateEnd"
/*@
   VecGhostUpdateEnd - End the vector scatter to update the vector from
   local representation to global or global representation to local.

   Neighbor-wise Collective on Vec

   Input Parameters:
+  g - the vector (obtained with VecCreateGhost() or VecDuplicate())
.  insertmode - one of ADD_VALUES or INSERT_VALUES
-  scattermode - one of SCATTER_FORWARD or SCATTER_REVERSE

   Notes:

   Use the following to update the ghost regions with correct values from the owning process
.vb
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Use the following to accumulate the ghost region values onto the owning processors
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
.ve

   To accumulate the ghost region values onto the owning processors and then update
   the ghost regions correctly, call the later followed by the former, i.e.,
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Level: advanced

.seealso: VecCreateGhost(), VecGhostUpdateBegin(), VecGhostGetLocalForm(),
          VecGhostRestoreLocalForm(),VecCreateGhostWithArray()

@*/
PetscErrorCode  VecGhostUpdateEnd(Vec g,InsertMode insertmode,ScatterMode scattermode)
{
  Vec_MPI        *v;
  PetscErrorCode ierr;
  PetscBool      ismpi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)g,VECMPI,&ismpi);CHKERRQ(ierr);
  if (ismpi) {
    v = (Vec_MPI*)g->data;
    if (!v->localrep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Vector is not ghosted");
    if (!v->localupdate) PetscFunctionReturn(0);
    if (scattermode == SCATTER_REVERSE) {
      ierr = VecScatterEnd(v->localupdate,v->localrep,g,insertmode,scattermode);CHKERRQ(ierr);
    } else {
      ierr = VecScatterEnd(v->localupdate,g,v->localrep,insertmode,scattermode);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGhostGetValues"
/*@
   VecGhostGetValues - get values available in local form via their global indices

   Not Collective

   Input Arguments:
+  G - global ghosted vector
.  n - number of indices
-  gidx - global indices at which to obtain values

   Output Arguments:
.  values - vector entries at locations gidx

   Level: advanced

   Notes:

   VecGhostUpdateBegin() and VecGhostUpdateEnd() must be called after modification of the global vector to ensure that
   this function returns current values of the global vector.

   If used in a performance-sensitive way, it is recommended to use VecGhostGetLocalForm() and VecGetValues() or
   VecGetArrayRead() to access entries by local index.  Mapping global indices to local indices involves a search (or
   non-scalable data structure) while local-to-global is simple array lookup.  This function exists for legacy
   application that must access vectors based on global index.

.seealso: VecGetValues(), VecGhostGetLocalForm(), VecGhostUpdateBegin(), VecGhostUpdateEnd()
@*/
PetscErrorCode VecGhostGetValues(Vec G,PetscInt n,const PetscInt gidx[],PetscScalar values[])
{
  PetscErrorCode ierr;
  Vec L;
  PetscInt i,bs,rstart,rend;
  const PetscScalar *l;
  Vec_MPI *g;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(G,VEC_CLASSID,1);
  g = (Vec_MPI*)G->data;
  ierr = VecGhostGetLocalForm(G,&L);CHKERRQ(ierr);
  ierr = VecGetBlockSize(G,&bs);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(G,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (rstart <= gidx[i] && gidx[i] < rend) {
      values[i] = l[gidx[i] - rstart];
    } else {
      PetscInt nghost = g->nghost / bs,loc;
      if (PetscUnlikely(!g->lsorted)) {
        PetscInt j;
        ISLocalToGlobalMapping ltog;
        ierr = PetscMalloc2(nghost,&g->lsorted,nghost,&g->gsorted);CHKERRQ(ierr);
        for (j=0; j<nghost; j++) {
          g->lsorted[j] = (rend-rstart)/bs + j;
        }
        ierr = VecGetLocalToGlobalMapping(G,&ltog);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingApplyBlock(ltog,nghost,g->lsorted,g->gsorted);CHKERRQ(ierr);
        ierr = PetscSortIntWithArray(nghost,g->gsorted,g->lsorted);CHKERRQ(ierr);
      }
      ierr = PetscFindInt(gidx[i]/bs,nghost,g->gsorted,&loc);CHKERRQ(ierr);
      if (loc < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Global index %D not a ghost point of vector",gidx[i]);CHKERRQ(ierr);
      values[i] = l[g->lsorted[loc]*bs + gidx[i]%bs];
    }
  }

  ierr = VecGhostRestoreLocalForm(G,&L);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetOption_MPI"
PetscErrorCode VecSetOption_MPI(Vec v,VecOption op,PetscBool flag)
{
  PetscFunctionBegin;
  if (op == VEC_IGNORE_OFF_PROC_ENTRIES)      v->stash.donotstash   = flag;
  else if (op == VEC_IGNORE_NEGATIVE_INDICES) v->stash.ignorenegidx = flag;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecResetArray_MPI"
PetscErrorCode VecResetArray_MPI(Vec vin)
{
  Vec_MPI        *v = (Vec_MPI*)vin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  if (v->localrep) {
    ierr = VecResetArray(v->localrep);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
