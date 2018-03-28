#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpacePminusView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_Pminus *pmin = (PetscSpace_Pminus *) sp->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer, "Reduced subspace of %D polynomials, %D-forms\n", sp->maxDegree, pmin->formDegree);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Pminus(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpacePminusView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Pminus(PetscSpace sp)
{
  PetscSpace_Pminus *pmin    = (PetscSpace_Pminus *) sp->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceDestroy(&pmin->pminus1);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&pmin->kplus1);CHKERRQ(ierr);
  ierr = PetscFree(pmin->ind);CHKERRQ(ierr);
  ierr = PetscFree(pmin->mul);CHKERRQ(ierr);
  ierr = PetscFree(pmin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Pminus(PetscSpace sp)
{
  PetscSpace_Pminus *pmin    = (PetscSpace_Pminus *) sp->data;
  PetscInt           Nc, Nv, k, Nk, Nkplus;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (pmin->setupCalled) PetscFunctionReturn(0);
  k = pmin->formDegree;
  Nc = sp->Nc;
  Nv = sp->Nv;
  /* get the number of forms */
  if (k < 0 || k >= Nv) SETERRQ2(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONG, "Form degree (%D) must be in [0,%D]\n", k, Nv);
  ierr = PetscDTBinomial_Internal(Nv,k,&Nk);CHKERRQ(ierr);
  if (Nc < Nk || (Nk % Nc)) SETERRQ2(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONG, "Number of components (%D) must be a multiple of number of forms (%D)\n", Nc, Nk);
  if (sp->degree < 0 && sp->maxDegree < 0) SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE, "One of degree (%D) and maxDegree (%D) must be non-negative\n", sp->degree, sp->maxDegree);
  if (sp->maxDegree >= 0) {
    sp->degree = sp->maxDegree - 1;
  } else {
    sp->maxDegree = sp->degree + 1;
  }
  if (k == 0 || k == Nv) {
    PetscInt deg = (k == 0) ? sp->degree : sp->degree - 1;

    ierr = PetscSpaceSetType(sp, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
    ierr = PetscSpaceSetNumComponents(sp, Nc);CHKERRQ(ierr);
    ierr = PetscSpaceSetNumVariables(sp, Nv);CHKERRQ(ierr);
    ierr = PetscSpaceSetDegree(sp, deg, deg);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);

    PetscFunctionReturn(0);
  }

  pmin->Nk = Nk;
  ierr = PetscDTBinomial_Internal(Nv,k + 1,&Nkplus);CHKERRQ(ierr);
  pmin->Nkplus = Nkplus;
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)sp), &pmin->pminus1);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(pmin->pminus1, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(pmin->pminus1, Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(pmin->pminus1, sp->degree-1, sp->degree-1);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(pmin->pminus1, 1);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(pmin->pminus1);CHKERRQ(ierr);

  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)sp), &pmin->kplus1);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(pmin->kplus1, PETSCSPACEPOLYMAX);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(pmin->kplus1, Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(pmin->kplus1, PETSC_DETERMINE, sp->degree-1);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(pmin->kplus1, 1);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(pmin->kplus1);CHKERRQ(ierr);
  pmin->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Pminus(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Pminus *pmin = (PetscSpace_Pminus *) sp->data;
  PetscInt           k, Nv, Nc;
  PetscInt           Np, deg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  Nv  = sp->Nv;
  k   = pmin->formDegree;
  Nc  = sp->Nc;
  deg = sp->maxDegree;
  ierr = PetscDTBinomial_Internal(Nv + deg, Nv, &Np);CHKERRQ(ierr); /* number of full polynomials */
  if ((Np * Nc * deg) % (deg + k)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not calculating Pminus dimension correctly");
  *dim = (Np * Nc * deg) / (deg + k);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Pminus(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Pminus *pmin  = (PetscSpace_Pminus *) sp->data;
  DM                 dm      = sp->dm;
  PetscInt           Nc      = sp->Nc;
  PetscInt           Nv      = sp->Nv;
  PetscInt           Nk      = pmin->Nk;
  PetscInt           Nkplus  = pmin->Nkplus;
  PetscInt           Niter   = pmin->Niter;
  PetscInt           Nf      = Nc / Nk;
  PetscReal         *pmB = NULL, *pmD = NULL, *pmH = NULL;
  PetscReal         *kpB = NULL, *kpD = NULL, *kpH = NULL;
  PetscInt           c, pdim, pmdim, kpdim, d, e, i, p;
  PetscReal         *intX;
  PetscInt         **ind = pmin->ind;
  PetscReal        **mul = pmin->mul;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!pmin->setupCalled) {ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);}
  ierr = PetscSpaceGetDimension(sp,&pdim);CHKERRQ(ierr);
  pdim /= Nf;
  ierr = PetscSpaceGetDimension(pmin->pminus1,&pmdim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(pmin->kplus1,&kpdim);CHKERRQ(ierr);
  if (B || D || H) {
    ierr = DMGetWorkArray(dm, npoints*pmdim,       MPIU_REAL, &pmB);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, npoints*kpdim,       MPIU_REAL, &kpB);CHKERRQ(ierr);
  }
  if (D || H) {
    ierr = DMGetWorkArray(dm, npoints*pmdim*Nv,    MPIU_REAL, &pmD);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, npoints*kpdim*Nv,    MPIU_REAL, &kpD);CHKERRQ(ierr);
  }
  if (H) {
    ierr = DMGetWorkArray(dm, npoints*pmdim*Nv*Nv, MPIU_REAL, &pmH);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, npoints*kpdim*Nv*Nv, MPIU_REAL, &pmH);CHKERRQ(ierr);
  }
  ierr = PetscSpaceEvaluate(pmin->pminus1, npoints, points, pmB, pmD, pmH);CHKERRQ(ierr);
  ierr = PetscSpaceEvaluate(pmin->kplus1,  npoints, points, kpB, kpD, kpH);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, Nk*Nkplus, MPIU_REAL, &intX);CHKERRQ(ierr);
  if (B) {
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pmdim; ++i) {
        for (c = 0; c < Nk; c++) {
          for (d = 0; d < Nk; d++) {
            B[(p*pdim*Nf + i*Nc)*Nc + c*Nc + d] = 0.;
          }
        }
        for (c = 0; c < Nk; ++c) {
          B[(p*pdim*Nf + i*Nc)*Nc + c*Nc + c] = pmB[p*pmdim + i];
        }
      }
      /* compute intX for this point */
      for (d = 0; d < Nkplus * Nk; d++) intX[d] = 0.;
      for (d = 0; d < Nv; d++) {
        for (c = 0; c < Niter; c++) {
          intX[ind[d][c]] = points[p * Nv + d] * mul[d][c];
        }
      }
      for (c = 0; c < Nkplus; c++) {
        for (d = 0; d < Nk; d++) {
        }
      }
      for (i = 0; i < kpdim; i++) {
        for (c = 0; c < Nkplus; c++) {
          for (d = 0; d < Nk; d++) {
            B[(p*pdim*Nf + pmdim*Nc + i*Nkplus*Nf)*Nc + c*Nc + d] = intX[c * Nk + d] * kpB[p*kpdim + i];
          }
        }
      }
    }
  }
  if (D) {
  }
  if (H) {
  }
  if (B && Nc > Nk) {
    /* Make direct sum basis for multicomponent space */
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        for (c = 1; c < Nc; ++c) {
          B[(p*pdim*Nc + i*Nc + c)*Nc + c] = B[(p*pdim + i)*Nc*Nc];
        }
      }
    }
  }
  if (D && Nc > Nk) {
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
  if (H && Nc > Nk) {
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
  ierr = DMRestoreWorkArray(dm, Nk*Nkplus, MPIU_REAL, &intX);CHKERRQ(ierr);
  if (H) {
    ierr = DMRestoreWorkArray(dm, npoints*kpdim*Nv*Nv, MPIU_REAL, &pmH);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, npoints*pmdim*Nv*Nv, MPIU_REAL, &pmH);CHKERRQ(ierr);
  }
  if (D || H) {
    ierr = DMRestoreWorkArray(dm, npoints*kpdim*Nv,    MPIU_REAL, &kpD);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, npoints*pmdim*Nv,    MPIU_REAL, &pmD);CHKERRQ(ierr);
  }
  if (B || D || H) {
    ierr = DMRestoreWorkArray(dm, npoints*kpdim,       MPIU_REAL, &kpB);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, npoints*pmdim,       MPIU_REAL, &pmB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Pminus(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions    = NULL;
  sp->ops->setup             = PetscSpaceSetUp_Pminus;
  sp->ops->view              = PetscSpaceView_Pminus;
  sp->ops->destroy           = PetscSpaceDestroy_Pminus;
  sp->ops->getdimension      = PetscSpaceGetDimension_Pminus;
  sp->ops->evaluate          = PetscSpaceEvaluate_Pminus;
  sp->ops->getheightsubspace = NULL;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPMINUS = "pminus" - A PetscSpace object that encapsulates the reduced polynomial space
                     for a set of alternating forms of fixed form degree.
                     Copies of forms are assumed to be identical.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Pminus(PetscSpace sp)
{
  PetscSpace_Pminus *pmin;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&pmin);CHKERRQ(ierr);
  sp->data = pmin;

  ierr = PetscSpaceInitialize_Pminus(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


