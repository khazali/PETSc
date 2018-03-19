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
  PetscInt           Ns, Nc, i, Nv, deg;
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
  PetscInt           Nv, Ns, i;
  PetscBool          uniform = PETSC_TRUE;
  PetscInt           deg, maxDeg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (tens->setupCalled) PetscFunctionReturn(0);
  ierr = PetscSpaceGetNumVariables(sp, &Nv);CHKERRQ(ierr);
  ierr = PetscSpaceTensorGetNumSubspaces(sp, &Ns);CHKERRQ(ierr);
  if (Ns == PETSC_DEFAULT) {
    Ns = Nv;
    ierr = PetscSpaceTensorSetNumSubspaces(sp, Ns);CHKERRQ(ierr);
  }
  if (!Ns) {
    if (Nv) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Cannot have zero subspaces");
  } else {
    PetscSpace s0;

    if (Nv > 0 && Ns > Nv) SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a tensor space with %D subspaces over %D variables\n", Ns, Nv);
    ierr = PetscSpaceTensorGetSubspace(sp, 0, &s0);CHKERRQ(ierr);
    for (i = 1; i < Ns; i++) {
      PetscSpace si;

      ierr = PetscSpaceTensorGetSubspace(sp, i, &si);CHKERRQ(ierr);
      if (si != s0) {uniform = PETSC_FALSE; break;}
    }
    if (uniform) {
      PetscInt   Nvs = Nv / Ns;

      if (Nv % Ns) SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONG,"Cannot use %D uniform subspaces for %D variable space\n", Ns, Nv);
      if (!s0) {ierr = PetscSpaceTensorCreateSubspace(sp, Nvs, &s0);CHKERRQ(ierr);}
      else     {ierr = PetscObjectReference((PetscObject) s0);CHKERRQ(ierr);}
      ierr = PetscSpaceSetUp(s0);CHKERRQ(ierr);
      for (i = 0; i < Ns; i++) {ierr = PetscSpaceTensorSetSubspace(sp, i, s0);CHKERRQ(ierr);}
      ierr = PetscSpaceDestroy(&s0);CHKERRQ(ierr);
    } else {
      for (i = 0 ; i < Ns; i++) {
        PetscSpace si;

        ierr = PetscSpaceTensorGetSubspace(sp, i, &si);CHKERRQ(ierr);
        if (!si) {ierr = PetscSpaceTensorCreateSubspace(sp, 1, &si);CHKERRQ(ierr);}
        else     {ierr = PetscObjectReference((PetscObject) si);CHKERRQ(ierr);}
        ierr = PetscSpaceSetUp(si);CHKERRQ(ierr);
        ierr = PetscSpaceTensorSetSubspace(sp, i, si);CHKERRQ(ierr);
        ierr = PetscSpaceDestroy(&si);CHKERRQ(ierr);
      }
    }
  }
  deg = PETSC_MAX_INT;
  maxDeg = 0;
  for (i = 0; i < Ns; i++) {
    PetscSpace si;
    PetscInt   iDeg, iMaxDeg;

    ierr = PetscSpaceTensorGetSubspace(sp, i, &si);CHKERRQ(ierr);
    ierr = PetscSpaceGetDegree(si, &iDeg, &iMaxDeg);CHKERRQ(ierr);
    deg    = PetscMin(deg, iDeg);
    maxDeg = PetscMax(maxDeg, iMaxDeg);
  }
  sp->degree    = deg;
  sp->maxDegree = maxDeg;
  tens->uniform = uniform;
  tens->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceDestroy_Tens(PetscSpace sp)
{
  PetscSpace_Tensor *tens    = (PetscSpace_Tensor *) sp->data;
  PetscInt           Ns, i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceTensorGetNumSubspaces(sp, &Ns);CHKERRQ(ierr);
  for (i = 0; i < Ns; i++) {ierr = PetscSpaceDestroy(&tens->spaces[i]);CHKERRQ(ierr);}
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorSetSubspace_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorGetSubspace_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorSetNumSubspaces_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorGetNumSubspaces_C", NULL);CHKERRQ(ierr);
  ierr = PetscFree(tens->spaces);CHKERRQ(ierr);
  ierr = PetscFree(tens);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceGetDimension_Tensor(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) sp->data;
  PetscInt           i, Ns, Nc, d;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (tens->dim == PETSC_DEFAULT) {
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    Ns = tens->numSpaces;
    Nc = sp->Nc;
    d  = 1;
    for (i = 0; i < Ns; i++) {
      PetscInt id;

      ierr = PetscSpaceGetDimension(tens->spaces[i], &id);CHKERRQ(ierr);
      d *= (id / Nc);
    }
    d *= Nc;
    tens->dim = d;
  }
  *dim = tens->dim;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceEvaluate_Tensor(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Tensor *tens  = (PetscSpace_Tensor *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         Nc      = sp->Nc;
  PetscInt         Nv      = sp->Nv;
  PetscInt         Ns;
  PetscReal       *lpoints, *tmp, *sB, *sD, *sH;
  PetscInt        *ind, *tup;
  PetscInt         c, pdim, d, e, der, der2, i, p, deg, o, s;
  PetscInt         pred;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!tens->setupCalled) {ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);}
  Ns = tens->numSpaces;
  pdim = tens->dim;
  ierr = DMGetWorkArray(dm, npoints*Nv, MPIU_REAL, &lpoints);CHKERRQ(ierr);
  if (B) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc, MPIU_REAL, &sB);CHKERRQ(ierr);}
  if (D) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*Nv, MPIU_REAL, &sD);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*Nv*Nv, MPIU_REAL, &sH);CHKERRQ(ierr);}
  for (s = 0, d = 0, pred = 1; s < Ns; s++) {
    PetscInt sd;

    ierr = PetscSpaceGetDimension(tens->spaces[s], &sd);CHKERRQ(ierr);
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < sd; i++) {
        lpoints[p * sd + i] = points[p*Nv + d + i];
      }
    }
    ierr = PetscSpaceEvaluate(tens->spaces[s], npoints, lpoints, sB, sD, sH);CHKERRQ(ierr);
    if (sB) {
    }
    d += sd;
    pred *= sd / Nc;
  }
  if (H) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*Nv*Nv, MPIU_REAL, &sH);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*Nv, MPIU_REAL, &sD);CHKERRQ(ierr);}
  if (B) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc, MPIU_REAL, &sB);CHKERRQ(ierr);}
  ierr = DMRestoreWorkArray(dm, npoints*Nv, MPIU_REAL, &lpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceTensorSetNumSubspaces_Tensor(PetscSpace space, PetscInt numSpaces)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;
  PetscInt           Ns;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (tens->setupCalled) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change number of subspaces after setup called\n");
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
  if (tens->setupCalled) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change subspace after setup called\n");
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
  tens->dim       = PETSC_DEFAULT;

  PetscFunctionReturn(0);
}

