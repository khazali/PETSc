#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_TBB"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_TBB(PetscThreadComm tcomm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_TBB"
PetscErrorCode PetscThreadCommRunKernel_TBB(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
