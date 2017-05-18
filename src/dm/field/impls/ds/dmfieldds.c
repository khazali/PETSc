#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/
#include <petscfe.h>
#include <petscdmplex.h>

typedef struct _n_DMField_DS
{
  PetscInt    fieldNum;
  Vec         vec;
  PetscObject disc;
  PetscBool   multifieldVec;
}
DMField_DS;

static PetscErrorCode DMFieldDestroy_DS(DMField field)
{
  DMField_DS     *dsfield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dsfield = (DMField_DS *) field->data;
  ierr = VecDestroy(&dsfield->vec);CHKERRQ(ierr);
  ierr = PetscObjectDereference(dsfield->disc);CHKERRQ(ierr);
  dsfield->disc = NULL;
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
  disc = dsfield->disc;
  if (iascii) {
    PetscViewerASCIIPrintf(viewer, "PetscDS field %D\n",dsfield->fieldNum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscObjectView(disc,viewer);CHKERRQ(ierr);
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

#define DMFieldDSdot(y,A,b,m,n,c,cast)                                           \
  do {                                                                           \
    PetscInt _i, _j, _k;                                                         \
    for (_i = 0; _i < (m); _i++) {                                               \
      for (_k = 0; _k < (c); _k++) {                                             \
        (y)[_i * (c) + _k] = 0.;                                                 \
      }                                                                          \
      for (_j = 0; _j < (n); _j++) {                                             \
        for (_k = 0; _k < (c); _k++) {                                           \
          (y)[_i * (c) + _k] += (A)[(_i * (n) + _j) * (c) + _k] * cast((b)[_j]); \
        }                                                                        \
      }                                                                          \
    }                                                                            \
  } while (0)

static PetscErrorCode DMFieldEvaluateFE_DS_Internal(DMField field, PetscInt numCells, const PetscInt *cells, PetscQuadrature quad, PetscDataType type, void *B, void *D, void *H)
{
  DMField_DS      *dsfield = (DMField_DS *) field->data;
  DM              dm;
  PetscObject     disc;
  PetscClassId    classid;
  PetscInt        nq, nc, dim, N;
  PetscSection    section;
  const PetscReal *qpoints;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
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
  /* TODO: batch */
  if (classid == PETSCFE_CLASSID) {
    PetscFE      fe = (PetscFE) disc;
    PetscInt     feDim, i;
    PetscReal    *fB = NULL, *fD = NULL, *fH = NULL;
    PetscInt     closureSize = 0;
    PetscScalar  *elem = NULL;

    ierr = PetscFEGetDimension(fe,&feDim);CHKERRQ(ierr);
    ierr = PetscFEGetTabulation(fe,nq,qpoints,B ? &fB : NULL,D ? &fD : NULL,H ? &fH : NULL);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm,feDim,PETSC_SCALAR,&elem);CHKERRQ(ierr);
    for (i = 0; i < numCells; i++) {
      PetscInt c = cells[i];

      ierr = DMPlexVecGetClosure(dm,section,dsfield->vec,c,&closureSize,&elem);CHKERRQ(ierr);
      if (B) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cB = &((PetscScalar *) B)[N * i];

          DMFieldDSdot(cB,fB,elem,nq,feDim,nc,(PetscScalar));
        } else {
          PetscReal *cB = &((PetscReal *) B)[N * i];

          DMFieldDSdot(cB,fB,elem,nq,feDim,nc,PetscRealPart);
        }
      }
      if (D) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cD = &((PetscScalar *) D)[N * dim * i];

          DMFieldDSdot(cD,fD,elem,nq,feDim,(nc * dim),(PetscScalar));
        } else {
          PetscReal *cD = &((PetscReal *) D)[N * dim * i];

          DMFieldDSdot(cD,fD,elem,nq,feDim,(nc * dim),PetscRealPart);
        }
      }
      if (H) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cH = &((PetscScalar *) H)[N * dim * dim * i];

          DMFieldDSdot(cH,fH,elem,nq,feDim,(nc * dim * dim),(PetscScalar));
        } else {
          PetscReal *cH = &((PetscReal *) H)[N * dim * dim * i];

          DMFieldDSdot(cH,fH,elem,nq,feDim,(nc * dim * dim),PetscRealPart);
        }
      }
    }
    ierr = DMPlexVecRestoreClosure(dm,section,dsfield->vec,-1,&closureSize,&elem);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,feDim,PETSC_SCALAR,&elem);CHKERRQ(ierr);
    ierr = PetscFERestoreTabulation(fe,nq,qpoints,B ? &fB : NULL,D ? &fD : NULL,H ? &fH : NULL);CHKERRQ(ierr);
  } else {SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented");}
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFE_DS(DMField field, PetscInt numCells, const PetscInt *cells, PetscQuadrature quad, PetscScalar *B, PetscScalar *D, PetscScalar *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMFieldEvaluateFE_DS_Internal(field,numCells,cells,quad,PETSC_SCALAR,B,D,H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFEReal_DS(DMField field, PetscInt numCells, const PetscInt *cells, PetscQuadrature quad, PetscReal *B, PetscReal *D, PetscReal *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMFieldEvaluateFE_DS_Internal(field,numCells,cells,quad,PETSC_REAL,B,D,H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldInitialize_DS(DMField field)
{
  PetscFunctionBegin;
  field->ops->destroy        = DMFieldDestroy_DS;
  field->ops->evaluate       = DMFieldEvaluate_DS;
  field->ops->evaluateReal   = DMFieldEvaluateReal_DS;
  field->ops->evaluateFE     = DMFieldEvaluateFE_DS;
  field->ops->evaluateFEReal = DMFieldEvaluateFEReal_DS;
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
  if (!disc) {
    PetscInt        cStart, cEnd, dim;
    PetscInt        localConeSize = 0, coneSize;
    PetscFE         fe;
    PetscDualSpace  Q;
    PetscSpace      P;
    DM              K;
    PetscBool       isSimplex;

    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    if (cEnd > cStart) {
      ierr = DMPlexGetConeSize(dm, cStart, &localConeSize);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&localConeSize,&coneSize,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    isSimplex = (coneSize == (dim + 1)) ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P);CHKERRQ(ierr);
    ierr = PetscSpaceSetNumComponents(P, numComponents);CHKERRQ(ierr);
    ierr = PetscSpacePolynomialSetNumVariables(P, dim);CHKERRQ(ierr);
    ierr = PetscSpacePolynomialSetTensor(P, isSimplex ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscSpaceSetOrder(P, 1);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateReferenceCell(Q, dim, isSimplex, &K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
    ierr = DMDestroy(&K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetNumComponents(Q, numComponents);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetOrder(Q, 1);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeSetTensor(Q, isSimplex ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), &fe);CHKERRQ(ierr);
    ierr = PetscFESetBasisSpace(fe, P);CHKERRQ(ierr);
    ierr = PetscFESetDualSpace(fe, Q);CHKERRQ(ierr);
    ierr = PetscFESetNumComponents(fe, numComponents);CHKERRQ(ierr);
    ierr = PetscFESetUp(fe);CHKERRQ(ierr);
    ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
    ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
    disc = (PetscObject) fe;
  } else {
    ierr = PetscObjectReference(disc);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
  if (id == PETSCFE_CLASSID) {
    PetscFE fe = (PetscFE) disc;

    ierr = PetscFEGetNumComponents(fe,&numComponents);CHKERRQ(ierr);
  } else {SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented");}
  ierr = DMFieldCreate(dm,numComponents,DMFIELD_VERTEX,&b);CHKERRQ(ierr);
  ierr = DMFieldSetType(b,DMFIELDDS);CHKERRQ(ierr);
  dsfield = (DMField_DS *) b->data;
  dsfield->fieldNum = fieldNum;
  dsfield->disc = disc;
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  dsfield->vec = vec;
  PetscFunctionReturn(0);
}
