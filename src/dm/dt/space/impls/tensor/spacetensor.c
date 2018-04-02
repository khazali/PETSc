#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceTensorCreateSubspace(PetscSpace space, PetscInt Nvs, PetscSpace *subspace)
{
  PetscInt    degree;
  const char *prefix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetDegree(space, &degree, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)space, &prefix);CHKERRQ(ierr);
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)space), subspace);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(*subspace, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(*subspace, Nvs);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(*subspace, 1);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(*subspace, degree, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)*subspace, prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)*subspace, "subspace_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetFromOptions_Tensor(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
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

static PetscErrorCode PetscSpaceView_Tensor(PetscSpace sp, PetscViewer viewer)
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

static PetscErrorCode PetscSpaceSetUp_Tensor(PetscSpace sp)
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
    maxDeg += iMaxDeg;
  }
  sp->degree    = deg;
  sp->maxDegree = maxDeg;
  tens->uniform = uniform;
  tens->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Tensor(PetscSpace sp)
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

static PetscErrorCode PetscSpaceGetDimension_Tensor(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) sp->data;
  PetscInt           i, Ns, Nc, d;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
  Ns = tens->numSpaces;
  Nc = sp->Nc;
  d  = 1;
  for (i = 0; i < Ns; i++) {
    PetscInt id;

    ierr = PetscSpaceGetDimension(tens->spaces[i], &id);CHKERRQ(ierr);
    d *= id;
  }
  d *= Nc;
  *dim = d;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Tensor(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Tensor *tens  = (PetscSpace_Tensor *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         Nc      = sp->Nc;
  PetscInt         Nv      = sp->Nv;
  PetscInt         Ns;
  PetscReal       *lpoints, *sB = NULL, *sD = NULL, *sH = NULL;
  PetscInt         c, pdim, d, e, der, der2, i, l, si, p, s, step;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!tens->setupCalled) {ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);}
  Ns = tens->numSpaces;
  ierr = PetscSpaceGetDimension(sp,&pdim);CHKERRQ(ierr);
  pdim /= Nc;
  ierr = DMGetWorkArray(dm, npoints*Nv, MPIU_REAL, &lpoints);CHKERRQ(ierr);
  if (B || D || H) {ierr = DMGetWorkArray(dm, npoints*pdim,       MPIU_REAL, &sB);CHKERRQ(ierr);}
  if (D || H)      {ierr = DMGetWorkArray(dm, npoints*pdim*Nv,    MPIU_REAL, &sD);CHKERRQ(ierr);}
  if (H)           {ierr = DMGetWorkArray(dm, npoints*pdim*Nv*Nv, MPIU_REAL, &sH);CHKERRQ(ierr);}
  if (B) {
    for (i = 0; i < npoints*pdim; i++) B[i * Nc*Nc] = 1.;
  }
  if (D) {
    for (i = 0; i < npoints*pdim; i++) {
      for (l = 0; l < Nv; l++) {
        D[i * Nc*Nc*Nv + l] = 1.;
      }
    }
  }
  if (H) {
    for (i = 0; i < npoints*pdim; i++) {
      for (l = 0; l < Nv*Nv; l++) {
        H[i * Nc*Nc*Nv*Nv + l] = 1.;
      }
    }
  }
  for (s = 0, d = 0, step = 1; s < Ns; s++) {
    PetscInt sNv, spdim;
    PetscInt skip, j, k;

    ierr = PetscSpaceGetNumVariables(tens->spaces[s], &sNv);CHKERRQ(ierr);
    ierr = PetscSpaceGetDimension(tens->spaces[s], &spdim);CHKERRQ(ierr);
    if ((pdim % step) || (pdim % spdim))  SETERRQ6(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Bad tensor loop: Nv %d, Ns %D, pdim %D, s %D, step %D, spdim %D", Nv, Ns, pdim, s, step, spdim);
    skip = pdim / (step * spdim);
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < sNv; i++) {
        lpoints[p * sNv + i] = points[p*Nv + d + i];
      }
    }
    ierr = PetscSpaceEvaluate(tens->spaces[s], npoints, lpoints, sB, sD, sH);CHKERRQ(ierr);
    if (B) {
      for (p = 0; p < npoints; p++) {
        for (k = 0; k < skip; k++) {
          for (si = 0; si < spdim; si++) {
            for (j = 0; j < step; j++) {
              i = (k * spdim + si) * step + j;
              B[(pdim * p + i) * Nc * Nc] *= sB[spdim * p + si];
            }
          }
        }
      }
    }
    if (D) {
      for (p = 0; p < npoints; p++) {
        for (k = 0; k < skip; k++) {
          for (si = 0; si < spdim; si++) {
            for (j = 0; j < step; j++) {
              i = (k * spdim + si) * step + j;
              for (der = 0; der < Nv; der++) {
                if (der >= d && der < d + sNv) {
                  D[(pdim * p + i) * Nc*Nc*Nv + der] *= sD[(spdim * p + si) * sNv + der - d];
                } else {
                  D[(pdim * p + i) * Nc*Nc*Nv + der] *= sB[spdim * p + si];
                }
              }
            }
          }
        }
      }
    }
    if (H) {
      for (p = 0; p < npoints; p++) {
        for (k = 0; k < skip; k++) {
          for (si = 0; si < spdim; si++) {
            for (j = 0; j < step; j++) {
              i = (k * spdim + si) * step + j;
              for (der = 0; der < Nv; der++) {
                for (der2 = 0; der2 < Nv; der2++) {
                  if (der >= d && der < d + sNv && der2 >= d && der2 < d + sNv) {
                    H[((pdim * p + i) * Nc*Nc*Nv + der) * Nv + der2] *= sH[((spdim * p + si) * sNv + der - d) * sNv + der2 - d];
                  } else if (der >= d && der < d + sNv) {
                    H[((pdim * p + i) * Nc*Nc*Nv + der) * Nv + der2] *= sD[(spdim * p + si) * sNv + der - d];
                  } else if (der2 >= d && der2 < d + sNv) {
                    H[((pdim * p + i) * Nc*Nc*Nv + der) * Nv + der2] *= sD[(spdim * p + si) * sNv + der2 - d];
                  } else {
                    H[((pdim * p + i) * Nc*Nc*Nv + der) * Nv + der2] *= sB[spdim * p + si];
                  }
                }
              }
            }
          }
        }
      }
    }
    d += sNv;
    step *= spdim;
  }
  if (B && Nc > 1) {
    /* Make direct sum basis for multicomponent space */
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        for (c = 1; c < Nc; ++c) {
          B[(p*pdim*Nc + i*Nc + c)*Nc + c] = B[(p*pdim + i)*Nc*Nc];
        }
      }
    }
  }
  if (D && Nc > 1) {
    /* Make direct sum basis for multicomponent space */
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        for (c = 1; c < Nc; ++c) {
          for (d = 0; d < Nv; ++d) {
            D[((p*pdim*Nc + i*Nc + c)*Nc + c)*Nv + d] = D[(p*pdim + i)*Nc*Nc*Nv + d];
          }
        }
      }
    }
  }
  if (H && Nc > 1) {
    /* Make direct sum basis for multicomponent space */
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        for (c = 1; c < Nc; ++c) {
          for (d = 0; d < Nv; ++d) {
            for (e = 0; e < Nv; ++e) {
              H[(((p*pdim*Nc + i*Nc + c)*Nc + c)*Nv + d)*Nv + e] = H[((p*pdim + i)*Nc*Nc*Nv + d)*Nv + e];
            }
          }
        }
      }
    }
  }
  if (H)           {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nv*Nv, MPIU_REAL, &sH);CHKERRQ(ierr);}
  if (D || H)      {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nv,    MPIU_REAL, &sD);CHKERRQ(ierr);}
  if (B || D || H) {ierr = DMRestoreWorkArray(dm, npoints*pdim,       MPIU_REAL, &sB);CHKERRQ(ierr);}
  ierr = DMRestoreWorkArray(dm, npoints*Nv, MPIU_REAL, &lpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceTensorSetNumSubspaces(PetscSpace sp, PetscInt numSpaces)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscSpaceTensorSetNumSubspaces_C",(PetscSpace,PetscInt),(sp,numSpaces));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceTensorGetNumSubspaces(PetscSpace sp, PetscInt *numSpaces)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidIntPointer(numSpaces, 2);
  ierr = PetscTryMethod(sp,"PetscSpaceTensorGetNumSubspaces_C",(PetscSpace,PetscInt*),(sp,numSpaces));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceTensorSetSubspace(PetscSpace sp, PetscInt s, PetscSpace subsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (subsp) PetscValidHeaderSpecific(subsp, PETSCSPACE_CLASSID, 3);
  ierr = PetscTryMethod(sp,"PetscSpaceTensorSetSubspace_C",(PetscSpace,PetscInt,PetscSpace),(sp,s,subsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceTensorGetSubspace(PetscSpace sp, PetscInt s, PetscSpace *subsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(subsp, 3);
  ierr = PetscTryMethod(sp,"PetscSpaceTensorGetSubspace_C",(PetscSpace,PetscInt,PetscSpace*),(sp,s,subsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceTensorSetNumSubspaces_Tensor(PetscSpace space, PetscInt numSpaces)
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

static PetscErrorCode PetscSpaceTensorGetNumSubspaces_Tensor(PetscSpace space, PetscInt *numSpaces)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;

  PetscFunctionBegin;
  *numSpaces = tens->numSpaces;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceTensorSetSubspace_Tensor(PetscSpace space, PetscInt s, PetscSpace subspace)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;
  PetscInt           Ns;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (tens->setupCalled) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change subspace after setup called\n");
  Ns = tens->numSpaces;
  if (Ns < 0) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceTensorSetNumSubspaces() first\n");
  if (s < 0 || s >= Ns) SETERRQ1(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D\n",subspace);
  ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&tens->spaces[s]);CHKERRQ(ierr);
  tens->spaces[s] = subspace;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceTensorGetSubspace_Tensor(PetscSpace space, PetscInt s, PetscSpace *subspace)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;
  PetscInt           Ns;

  PetscFunctionBegin;
  Ns = tens->numSpaces;
  if (Ns < 0) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceTensorSetNumSubspaces() first\n");
  if (s < 0 || s >= Ns) SETERRQ1(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D\n",subspace);
  *subspace = tens->spaces[s];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Tensor(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Tensor;
  sp->ops->setup             = PetscSpaceSetUp_Tensor;
  sp->ops->view              = PetscSpaceView_Tensor;
  sp->ops->destroy           = PetscSpaceDestroy_Tensor;
  sp->ops->getdimension      = PetscSpaceGetDimension_Tensor;
  sp->ops->evaluate          = PetscSpaceEvaluate_Tensor;
  sp->ops->getheightsubspace = NULL;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorGetNumSubspaces_C", PetscSpaceTensorGetNumSubspaces_Tensor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorSetNumSubspaces_C", PetscSpaceTensorSetNumSubspaces_Tensor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorGetSubspace_C", PetscSpaceTensorGetSubspace_Tensor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorSetSubspace_C", PetscSpaceTensorSetSubspace_Tensor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACETENSOR = "tensor" - A PetscSpace object that encapsulates a tensor product space.
                     Subspaces are scalar spaces (num of componenents = 1), vector components
                     are assumed to be identical.

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

  ierr = PetscSpaceInitialize_Tensor(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
