#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceTensorCreateSubspace(PetscSpace space, PetscInt Nvs, PetscSpace *subspace)
{
  PetscInt Nc, degree;
  const char *prefix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumComponents(space, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(space, &degree, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)space, &prefix);CHKERRQ(ierr);
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)space), subspace);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(*subspace, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(*subspace, Nvs);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(*subspace, Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(*subspace, degree);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)*subspace, prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)*subspace, "subspace_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSetFromOptions_Tensor(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) sp->data;
  PetscInt           Ns, Nc, i, Nv, deg, Nvs;
  PetscBool          uniform = PETSC_TRUE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumVariables(sp, &Nv);CHKERRQ(ierr);
  if (!Nv) PetscFunctionReturn(0);
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceTensorGetNumSubspaces(sp, &Ns);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &deg, NULL);CHKERRQ(ierr);
  if (Ns > 1) {
    PetscSpace s0;

    ierr = PetscSpaceTensorGetSubspace(sp, 0, &s0);CHKERRQ(ierr);
    for (i = 1; i < Ns; i++) {
      PetscSpace si;

      ierr = PetscSpaceTensorGetSubspace(sp, i, &si);CHKERRQ(ierr);
      if (si != s0) {uniform = PETSC_FALSE; break;}
    }
  }
  Ns = (Ns == PETSC_DEFAULT) ? PetscMax(Nv,1) : Ns;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace tensor options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_tensor_spaces", "The number of subspaces", "PetscSpaceTensorSetNumSubspaces", Ns, &Ns, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_tensor_uniform", "Subspaces are identical", "PetscSpaceTensorSetFromOptions", uniform, &uniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (Ns < 0 || (Nv > 0 && Ns == 0)) SETERRQ1(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a tensor space made up of %D spaces\n",Ns);
  if (Nv > 0 && Ns > Nv) SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a tensor space with %D subspaces over %D variables\n", Ns, Nv);
  if (Ns != tens->numSpaces) {ierr = PetscSpaceTensorSetNumSubspaces(sp, Ns);CHKERRQ(ierr);}
  if (uniform) {
    PetscInt   Nvs = Nv / Ns;
    PetscSpace subspace;

    if (Nv % Ns) SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONG,"Cannot use %D uniform subspaces for %D variable space\n", Ns, Nv);
    ierr = PetscSpaceTensorGetSubspace(sp, 0, &subspace);CHKERRQ(ierr);
    if (!subspace) {ierr = PetscSpaceTensorCreateSubspace(sp, Nvs, &subspace);CHKERRQ(ierr);}
    else           {ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);}
    ierr = PetscSpaceSetFromOptions(subspace);CHKERRQ(ierr);
    for (i = 0; i < Ns; i++) {ierr = PetscSpaceTensorSetSubspace(sp, i, subspace);CHKERRQ(ierr);}
    ierr = PetscSpaceDestroy(&subspace);CHKERRQ(ierr);
  } else {
    for (i = 0; i < Ns; i++) {
      PetscSpace subspace;

      ierr = PetscSpaceTensorGetSubspace(sp, i, &subspace);CHKERRQ(ierr);
      if (!subspace) {
        char tprefix[128];

        ierr = PetscSpaceTensorCreateSubspace(sp, 1, &subspace);CHKERRQ(ierr);
        ierr = PetscSNPrintf(tprefix, 128, "%d_",(int)i);CHKERRQ(ierr);
        ierr = PetscObjectAppendOptionsPrefix((PetscObject)subspace, tprefix);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
      }
      ierr = PetscSpaceSetFromOptions(subspace);CHKERRQ(ierr);
      ierr = PetscSpaceTensorSetSubspace(sp, i, subspace);CHKERRQ(ierr);
      ierr = PetscSpaceDestroy(&subspace);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceTensorView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) sp->data;
  PetscBool          uniform = PETSC_TRUE;
  PetscInt           Ns = tens->numSpaces, i, n;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  for (i = 1; i < Ns; i++) {
    if (tens->spaces[i] != tens->spaces[0]) {uniform = PETSC_FALSE; break;}
  }
  if (uniform) {ierr = PetscViewerASCIIPrintf(viewer, "Tensor space of %D subspaces (all identical)\n", Ns);CHKERRQ(ierr);
  } else       {ierr = PetscViewerASCIIPrintf(viewer, "Tensor space of %D subspaces\n", Ns);CHKERRQ(ierr);}
  n = uniform ? 1 : Ns;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    ierr = PetscSpaceView(tens->spaces[i], viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceView_Tensor(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpaceTensorView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSetUp_Tensor(PetscSpace sp)
{
  PetscSpace_Tensor *tens    = (PetscSpace_Tensor *) sp->data;
  PetscInt           ndegree = sp->degree+1;
  PetscInt           deg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
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

