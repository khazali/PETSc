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
. reduced_type - the method TAO is using for subsetting (TAO_SUBSET_SUBVEC, TAO_SUBSET_MASK,  TAO_SUBSET_MATRIXFREE)
- maskvalue - the value to set the unused vector elements to (for TAO_SUBSET_MASK or TAO_SUBSET_MATRIXFREE)

  Output Parameters:
. vreduced - the subvector

  Notes:
  maskvalue should usually be 0.0, unless a pointwise divide will be used.

@*/
PetscErrorCode TaoVecGetSubVec(Vec vfull, IS is, TaoSubsetType reduced_type, PetscScalar maskvalue, Vec *vreduced)
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
    switch (reduced_type) {
    case TAO_SUBSET_SUBVEC:
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
      break;

    case TAO_SUBSET_MASK:
    case TAO_SUBSET_MATRIXFREE:
      /* vr[i] = vf[i]   if i in is
       vr[i] = 0       otherwise */
      if (*vreduced == NULL) {
        ierr = VecDuplicate(vfull,vreduced);CHKERRQ(ierr);
      }
      CHKMEMQ;
      ierr = VecSet(*vreduced,maskvalue);CHKERRQ(ierr);
      CHKMEMQ;
      ierr = ISGetLocalSize(is,&nlocal);CHKERRQ(ierr);
      CHKMEMQ;
      ierr = VecGetOwnershipRange(vfull,&flow,&fhigh);CHKERRQ(ierr);
      CHKMEMQ;
      ierr = VecGetArray(vfull,&fv);CHKERRQ(ierr);
      CHKMEMQ;
      ierr = VecGetArray(*vreduced,&rv);CHKERRQ(ierr);
      CHKMEMQ;
      ierr = ISGetIndices(is,&s);CHKERRQ(ierr);
      if (nlocal > (fhigh-flow)) SETERRQ2(PETSC_COMM_WORLD,1,"IS local size %d > Vec local size %d",nlocal,fhigh-flow);
      for (i=0;i<nlocal;i++) {
        if (0) {printf("setting rv[%d] = fv[%d]\n",s[i]-flow,s[i]-flow);}
        rv[s[i]-flow] = fv[s[i]-flow];
      }
      CHKMEMQ;
      ierr = ISRestoreIndices(is,&s);CHKERRQ(ierr);
      CHKMEMQ;
      ierr = VecRestoreArray(vfull,&fv);CHKERRQ(ierr);
      CHKMEMQ;

      ierr = VecRestoreArray(*vreduced,&rv);CHKERRQ(ierr);
      CHKMEMQ;
      break;
    }
  }
  PetscFunctionReturn(0);
}
