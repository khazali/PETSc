#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

PetscErrorCode PetscSpaceSetFromOptions_Tensor(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Tens *tens = (PetscSpace_Tens *) sp->data;
  PetscInt         Ns, Nc, i, Nv, deg;
  PetscBool        uniform = PETSC_TRUE;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumVariables(sp, &Nv);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceTensorGetNumSubspaces(sp, &Ns);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &deg, NULL);CHKERRQ(ierr);
  Ns = (Ns == PETSC_DEFAULT) ? Nv : Ns;
  if (Ns > 1) {
    PetscSpace s0;

    ierr = PetscSpaceTensorGetSubspace(sp, 0, &s0);CHKERRQ(ierr);
    for (i = 1; i < Ns; i++) {
      PetscSpace si;

      ierr = PetscSpaceTensorGetSubspace(sp, i, &si);CHKERRQ(ierr);

      if (si != s0) {
        uniform = PETSC_FALSE;
        break;
      }
    }
  }
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace tensor options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_tens_spaces", "The number of subspaces", "PetscSpaceTensorSetNumSubspaces", Ns, &Ns, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_tens_uniform", "Subspaces are identical", "PetscSpaceTensorSetFromOptions", uniform, &uniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (Ns != tens->numSpaces) {ierr = PetscSpaceTensorSetNumSubspaces(space, Ns);CHKERRQ(ierr);}
  for (i = 0; i < Ns; i++) {
    PetscSpace subspace;

    ierr = PetscSpaceTensorGetSubspace(space, i, &subspace);CHKERRQ(ierr);
    if (!subspace) {
      char tprefix[128];

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)space), &subspace);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(subspace, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumVariable(subspace, 1);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(subspace, Nc);CHKERRQ(ierr);
      ierr = PetscSpaceSetDegree(subspace, deg);CHKERRQ(ierr);
      ierr = PetscSNPrintf(tprefix, 128, "sub_%d_",(int)i);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)subspace, tprefix);CHKERRQ(ierr);
      ierr = PetscSpaceSetFromOptions(subspace);CHKERRQ(ierr);
      ierr = PetscSpaceTensorSetSubspace(space, i, subspace);CHKERRQ(ierr);
      ierr = PetscSpaceDestroy
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceTensorSetNumSubspaces_Tensor(PetscSpace space, PetscInt numSpaces)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;
  PetscInt           Ns;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  Ns = tens->numSpaces;
  if (numSpaces == Ns) PetscFunctionReturn(0);
  if (Ns >= 0) {
    PetscInt s;

    for (s = 0; s < Ns; s++) {ierr = PetscSpaceDestroy(&tens->spaces[s]);CHKERRQ(ierr);}
    ierr = PetscFree(tens->spaces);CHKERRQ(ierr);
  }
  Ns = tens->numSpaces = numSpaces;
  ierr = PetscCalloc1(Ns, &tens->spaces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceTensorGetNumSubspaces_Tensor(PetscSpace space, PetscInt *numSpaces)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;

  PetscFunctionBegin;
  *numSpaces = tens->numSpaces;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceTensorSetSubspace_Tensor(PetscSpace space, PetscInt s, PetscSpace subspace)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;
  PetscInt           Ns;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  Ns = tens->numSpaces;
  if (Ns < 0) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceTensorSetNumSubspaces() first\n");
  if (s < 0) SETERRQ1(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid negative subspace number %D\n",subspace);
  ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&tens->spaces[s]);CHKERRQ(ierr);
  tens->spaces[s] = subspace;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceTensorGetSubspace_Tensor(PetscSpace space, PetscInt s, PetscSpace *subspace)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;
  PetscInt           Ns;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  Ns = tens->numSpaces;
  if (Ns < 0) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceTensorSetNumSubspaces() first\n");
  if (s < 0) SETERRQ1(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid negative subspace number %D\n",subspace);
  *subspace = tens->spaces[s];
  PetscFunctionReturn(0);
}


/*MC
  PETSCSPACETENSOR = "tensor" - A PetscSpace object that encapsulates a tensor product space.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Tensor(PetscSpace sp)
{
  PetscSpace_Tensor *tens;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&tens);CHKERRQ(ierr);
  sp->data = tens;

  tens->numSpaces = PETSC_DEFAULT;

  PetscFunctionReturn(0);
}

