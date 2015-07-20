#include <petsctao.h> /*I "petsctao.h" I*/
#include <petsc/private/taoimpl.h>
#include <petsc/private/matimpl.h>

#undef __FUNCT__
#define __FUNCT__ "TaoVecGetSubVec"
/*@C
  TaoVecGetSubVec - Gets a subvector using the IS

  Input Parameters:
+ vfull - the full matrix
. is - the index set for the subvector
. usemask - if true then don't take actual subvector, copy to new vector and set
            values of indices not in the is to the maskvalue
- maskvalue - the value to set the unused vector elements to (if usemask is True)

  Output Parameters:
. vreduced - the subvector

  Notes:
  maskvalue should usually be 0.0, unless a pointwise divide will be used.

@*/
PetscErrorCode TaoVecGetSubVec(Vec vfull, IS is, PetscBool usemask, PetscScalar maskvalue, Vec *vreduced)
{
  PetscErrorCode ierr;
  PetscInt       nfull,nreduced,nreduced_local,rlow,rhigh,flow,fhigh;
  PetscInt       i,nlocal;
  PetscScalar    *fv,*rv;
  const PetscInt *s;
  IS             ident;
  VecType        vtype;
  VecScatter     scatter;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vfull,VEC_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);

  ierr = VecGetSize(vfull, &nfull);CHKERRQ(ierr);
  ierr = ISGetSize(is, &nreduced);CHKERRQ(ierr);

  if (nreduced == nfull) {
    ierr = VecDestroy(vreduced);CHKERRQ(ierr);
    ierr = VecDuplicate(vfull,vreduced);CHKERRQ(ierr);
    ierr = VecCopy(vfull,*vreduced);CHKERRQ(ierr);
  } else {
    if (usemask) {
      /* vr[i] = vf[i]   if i in is
       vr[i] = 0       otherwise */
      if (*vreduced == NULL) {
        ierr = VecDuplicate(vfull,vreduced);CHKERRQ(ierr);
      }
      ierr = VecSet(*vreduced,maskvalue);CHKERRQ(ierr);
      ierr = ISGetLocalSize(is,&nlocal);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(vfull,&flow,&fhigh);CHKERRQ(ierr);
      ierr = VecGetArray(vfull,&fv);CHKERRQ(ierr);
      ierr = VecGetArray(*vreduced,&rv);CHKERRQ(ierr);
      ierr = ISGetIndices(is,&s);CHKERRQ(ierr);
      if (nlocal > (fhigh-flow)) SETERRQ2(PETSC_COMM_WORLD,1,"IS local size %d > Vec local size %d",nlocal,fhigh-flow);
      for (i=0;i<nlocal;i++) {
        if (0) {printf("setting rv[%d] = fv[%d]\n",s[i]-flow,s[i]-flow);}
        rv[s[i]-flow] = fv[s[i]-flow];
      }
      ierr = ISRestoreIndices(is,&s);CHKERRQ(ierr);
      ierr = VecRestoreArray(vfull,&fv);CHKERRQ(ierr);
      ierr = VecRestoreArray(*vreduced,&rv);CHKERRQ(ierr);
    } else {
      ierr = VecGetType(vfull,&vtype);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(vfull,&flow,&fhigh);CHKERRQ(ierr);
      ierr = ISGetLocalSize(is,&nreduced_local);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject)vfull,&comm);CHKERRQ(ierr);
      if (*vreduced) {
        ierr = VecDestroy(vreduced);CHKERRQ(ierr);
      }
      ierr = VecCreate(comm,vreduced);CHKERRQ(ierr);
      ierr = VecSetType(*vreduced,vtype);CHKERRQ(ierr);
      ierr = VecSetSizes(*vreduced,nreduced_local,nreduced);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(*vreduced,&rlow,&rhigh);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,nreduced_local,rlow,1,&ident);CHKERRQ(ierr);
      ierr = VecScatterCreate(vfull,is,*vreduced,ident,&scatter);CHKERRQ(ierr);
      ierr = VecScatterBegin(scatter,vfull,*vreduced,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(scatter,vfull,*vreduced,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
      ierr = ISDestroy(&ident);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
