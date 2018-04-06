#include <petsc/private/taomeritimpl.h>
#include <../src/tao/merit/impls/obj/obj.h>

PETSC_EXTERN PetscErrorCode TaoMeritCreate_Obj(TaoMerit merit)
{
  PetscErrorCode   ierr;
  TaoMerit_Obj     *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls,TAOMERIT_CLASSID,1);
  ierr = PetscNewLog(ls,&ctx);CHKERRQ(ierr);
  merit->data = (void*)ctx;
  merit->ops->computevalue = TaoMeritComputeValue_Obj;
  merit->ops->computedirderiv = TaoMeritComputeDirDeriv_Obj;
  merit->ops->computeall = TaoMeritComputeAll_Obj;
  merit->ops->destroy=TaoMeritDestroy_Obj;
  merit->ops->setfromoptions=TaoMeritSetFromOptions_Obj;
  PetscFunctionReturn(0);
}