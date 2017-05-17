#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/
#include <petscfe.h>
#include <petscdmplex.h>

typedef struct _n_DMField_DS
{
  PetscInt  fieldNum;
  Vec       vec;
  PetscBool multifieldVec;
}
DMField_DS;

static PetscErrorCode DMFieldDestroy_DS(DMField field)
{
  DMField_DS     *dsfield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dsfield = (DMField_DS *) field->data;
  ierr = VecDestroy(&dsfield->vec);CHKERRQ(ierr);
  ierr = PetscFree(dsfield);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldView_DS(DMField field,PetscViewer viewer)
{
  DMField_DS     *dsfield = (DMField_DS *) field->data;
  PetscBool      iascii;
  PetscDS        ds;
  PetscObject    disc;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm   = field->dm;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
  ierr = DMGetField(dm,dsfield->fieldNum,&disc);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerASCIIPrintf(viewer, "PetscDS field %D\n",dsfield->fieldNum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (disc) {
      ierr = PetscObjectView(disc,viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "Implicit discretization\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (dsfield->multifieldVec) {
    SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"View of subfield not implemented yet");
  } else {
    ierr = VecView(dsfield->vec,viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluate_DS(DMField field, Vec points, PetscScalar *B, PetscScalar *D, PetscScalar *H)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented yet");
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateReal_DS(DMField field, Vec points, PetscScalar *B, PetscScalar *D, PetscScalar *H)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented yet");
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFE_DS(DMField field, PetscInt numCells, const PetscInt *cells, PetscQuadrature quad, PetscScalar *B, PetscScalar *D, PetscScalar *H)
{
  DMField_DS      *dsfield = (DMField_DS *) field->data;
  DM              dm;
  PetscObject     disc;
  PetscClassId    classid;
  PetscInt        nq, nc, dim, N;
  PetscSection    section;
  const PetscReal *qpoints;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  dm   = field->dm;
  nc   = field->numComponents;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMGetField(dm,dsfield->fieldNum,&disc);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
  if (dsfield->multifieldVec) {
    ierr = PetscSectionGetField(section,dsfield->fieldNum,&section);CHKERRQ(ierr);
  }
  if (!disc) SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented");
  ierr = PetscObjectGetClassId(disc,&classid);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad,NULL,NULL,&nq,&qpoints,NULL);CHKERRQ(ierr);
  N = nc * nq;
  if (classid == PETSCFE_CLASSID) {
    PetscFE fe = (PetscFE) disc;
    PetscInt  feDim, i;
    PetscReal *fB = NULL, *fD = NULL, *fH = NULL;
    PetscInt closureSize = 0;
    PetscScalar *elem = NULL;

    ierr = PetscFEGetDimension(fe,&feDim);CHKERRQ(ierr);
    ierr = PetscFEGetTabulation(fe,nq,qpoints,B ? &fB : NULL,D ? &fD : NULL,H ? &fH : NULL);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm,feDim,PETSC_SCALAR,&elem);CHKERRQ(ierr);
    for (i = 0; i < numCells; i++) {
      PetscInt c = cells[i];

      ierr = DMPlexVecGetClosure(dm,section,dsfield->vec,c,&closureSize,&elem);CHKERRQ(ierr);
      if (B) {
        PetscScalar *cB = &B[N * i];
        PetscInt j, k, l;

        for (j = 0; j < nq; j++) {
          for (l = 0; l < nc; l++) {
            cB[nc * j + l] = 0.;
          }
          for (k = 0; k < feDim; k++) {
            for (l = 0; l < nc; l++) {
              cB[nc * j + l] += fB[(j * feDim + k) * nc + l] * elem[k];
            }
          }
        }
      }
      if (D) {
        PetscScalar *cD = &D[N * dim * i];
        PetscInt j, k, l;

        for (j = 0; j < nq; j++) {
          for (l = 0; l < nc * dim; l++) {
            cD[nc * dim * j + l] = 0.;
          }
          for (k = 0; k < feDim; k++) {
            for (l = 0; l < nc * dim; l++) {
              cD[nc * dim * j + l] += fD[(j * feDim + k) * nc * dim + l] * elem[k];
            }
          }
        }
      }
      if (H) {
        PetscScalar *cH = &H[N * dim * dim * i];
        PetscInt j, k, l;

        for (j = 0; j < nq; j++) {
          for (l = 0; l < nc * dim * dim; l++) {
            cH[nc * dim * dim * j + l] = 0.;
          }
          for (k = 0; k < feDim; k++) {
            for (l = 0; l < nc * dim * dim; l++) {
              cH[nc * dim * dim * j + l] += fH[(j * feDim + k) * nc * dim * dim + l] * elem[k];
            }
          }
        }
      }
    }
    ierr = DMPlexVecRestoreClosure(dm,section,dsfield->vec,-1,&closureSize,&elem);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,feDim,PETSC_SCALAR,&elem);CHKERRQ(ierr);
    ierr = PetscFERestoreTabulation(fe,nq,qpoints,B ? &fB : NULL,D ? &fD : NULL,H ? &fH : NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldInitialize_DS(DMField field)
{
  PetscFunctionBegin;
  field->ops->destroy        = DMFieldDestroy_DS;
  field->ops->evaluate       = DMFieldEvaluate_DS;
  field->ops->evaluateReal   = DMFieldEvaluateReal_DS;
  field->ops->evaluateFE     = DMFieldEvaluateFE_DS;
  //field->ops->evaluateFEReal = DMFieldEvaluateFEReal_DA;
  //field->ops->evaluateFV     = DMFieldEvaluateFV_DA;
  //field->ops->evaluateFVReal = DMFieldEvaluateFVReal_DA;
  field->ops->view           = DMFieldView_DS;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMFieldCreate_DS(DMField field)
{
  DMField_DS     *dsfield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(field,&dsfield);CHKERRQ(ierr);
  field->data = dsfield;
  ierr = DMFieldInitialize_DS(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldCreateDS(DM dm, PetscInt fieldNum, Vec vec,DMField *field)
{
  DMField        b;
  DMField_DS     *dsfield;
  PetscObject    disc;
  PetscClassId   id;
  PetscInt       numComponents = -1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetField(dm,fieldNum,&disc);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
  if (id == PETSCFE_CLASSID) {
    PetscFE fe = (PetscFE) disc;

    ierr = PetscFEGetNumComponents(fe,&numComponents);CHKERRQ(ierr);
  } else {SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented");}
  ierr = DMFieldCreate(dm,numComponents,DMFIELD_VERTEX,&b);CHKERRQ(ierr);
  ierr = DMFieldSetType(b,DMFIELDDS);CHKERRQ(ierr);
  dsfield = (DMField_DS *) b->data;
  dsfield->fieldNum = fieldNum;
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  dsfield->vec = vec;
  PetscFunctionReturn(0);
}
