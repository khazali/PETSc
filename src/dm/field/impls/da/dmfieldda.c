#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/
#include <petsc/private/dmimpl.h> /*I "petscdm.h" I*/
#include <petscdmda.h>

typedef struct _n_DMField_DA
{
  PetscScalar     *cornerVals;
  PetscReal       coordRange[3][2];
}
DMField_DA;

static PetscErrorCode DMFieldDestroy_DA(DMField field)
{
  DMField_DA     *dafield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dafield = (DMField_DA *) field->data;
  ierr = PetscFree(dafield->cornerVals);CHKERRQ(ierr);
  ierr = PetscFree(dafield);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldView_DA(DMField field,PetscViewer viewer)
{
  DMField_DA     *dafield = (DMField_DA *) field->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscInt i, c, dim;
    PetscInt nc;
    DM       dm = field->dm;

    PetscViewerASCIIPrintf(viewer, "Field corner values:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    nc = field->numComponents;
    for (i = 0, c = 0; i < (1 << dim); i++) {
      PetscInt j;

      for (j = 0; j < nc; j++, c++) {
        PetscScalar val = dafield->cornerVals[nc * i + j];

#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%g ",(double) val);CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%g+i%g ",(double) PetscRealPart(val),(double) PetscImaginaryPart(val));CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static void MultilinearEvaluate(PetscInt dim, PetscReal (*coordRange)[2], PetscInt nc, const PetscScalar *cv, PetscInt nPoints, const PetscScalar *points, PetscScalar *B, PetscScalar *D, PetscScalar *H)
{
  PetscInt i, j, k, l, m, p;

  PetscFunctionBeginHot;
  for (i = 0; i < nPoints; i++) {
    const PetscScalar *point = &points[dim * i];
    PetscReal eta[3] = {0.};

    for (j = 0; j < dim; j++) {
      eta[j] = (point[j] - coordRange[j][0]) / coordRange[j][1];
    }
    if (B) {
      PetscScalar *out = &B[nc * i];

      for (k = 0; k < nc; k++) out[k] = 0.;
      for (l = 0; l < (1 << dim); l++) {
        PetscReal w = 1.;

        for (j = 0; j < dim; j++) {
          w *= (l & (1 << j)) ? eta[j] : (1. - eta[j]);
        }
        for (k = 0; k < nc; k++) {
          out[k] += w * cv[nc * l + k];
        }
      }
    }
    if (D) {
      PetscScalar *out = &D[nc * dim * i];

      for (m = 0; m < nc * dim; m++) out[m] = 0.;
      for (l = 0; l < (1 << dim); l++) {
        for (m = 0; m < dim; m++) {
          PetscReal w = 1.;

          for (j = 0; j < dim; j++) {
            w *= (l & (1 << j)) ? ((j == m) ? 1./coordRange[j][1] : eta[j]) : ((j == m) ? -1./coordRange[j][1] : (1. - eta[j]));
          }
          for (k = 0; k < nc; k++) {
            out[k * dim + m] += w * cv[nc * l + k];
          }
        }
      }
    }
    if (H) {
      PetscScalar *out = &H[nc * dim * dim * i];

      for (m = 0; m < nc * dim * dim; m++) out[m] = 0.;
      for (l = 0; l < (1 << dim); l++) {
        for (m = 0; m < dim; m++) {
          for (p = m + 1; p < dim; p++) {
            PetscReal w = 1.;
            PetscInt q;

            q = 3 - m - p;
            for (j = 0; j < dim; j++) {
              w *= (l & (1 << j)) ? ((j != q) ? 1./coordRange[j][1] : eta[j]) : ((j != q) ? -1./coordRange[j][1] : (1. - eta[j]));
            }
            for (k = 0; k < nc; k++) {
              out[k * dim * dim + m * dim + p] += w * cv[nc * l + k];
              out[k * dim * dim + p * dim + m] += w * cv[nc * l + k];
            }
          }
        }
      }
    }
  }
  PetscFunctionReturnVoid();
}

static PetscErrorCode DMFieldEvaluate_DA(DMField field, Vec points, PetscScalar *B, PetscScalar *D, PetscScalar *H)
{
  DM             dm;
  DMField_DA     *dafield;
  PetscInt       dim;
  PetscInt       N, n, nc;
  const PetscScalar *array;
  PetscReal (*coordRange)[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm      = field->dm;
  nc      = field->numComponents;
  dafield = (DMField_DA *) field->data;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(points,&N);CHKERRQ(ierr);
  if (N % dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Point vector size %D not divisible by coordinate dimension %D\n",N,dim);
  n = N / dim;
  coordRange = &(dafield->coordRange[0]);
  ierr = VecGetArrayRead(points,&array);CHKERRQ(ierr);
  MultilinearEvaluate(dim,coordRange,nc,dafield->cornerVals,n,array,B,D,H);
  ierr = VecRestoreArrayRead(points,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateReal_DA(DMField field, Vec points, PetscReal *B, PetscReal *D, PetscReal *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ierr = DMFieldEvaluate_DA(field,points,B,D,H);CHKERRQ(ierr);
#else
  {
    DM          dm = field->dm;
    PetscInt    dim, N, n, i;
    PetscScalar *sB = NULL, *sD = NULL, *sH = NULL;
    nc = field->numComponents;
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    ierr = VecGetLocalSize(points,&N);CHKERRQ(ierr);
    if (N % dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Point vector size %D not divisible by coordinate dimension %D\n",N,dim);
    n = N / dim;
    if (B) {ierr = DMGetWorkArray(dm,n*nc,PETSC_SCALAR,&sB);CHKERRQ(ierr);}
    if (D) {ierr = DMGetWorkArray(dm,n*dim*nc,PETSC_SCALAR,&sD);CHKERRQ(ierr);}
    if (H) {ierr = DMGetWorkArray(dm,n*dim*dim*nc,PETSC_SCALAR,&sH);CHKERRQ(ierr);}
    ierr = DMFieldEvaluate_DA(field,points,sB,sD,sH);CHKERRQ(ierr);
    if (H) {
      for (i = 0; i < n * dim * dim * nc; i++) {H[i] = PetscRealPart(sH[i]);}
      ierr = DMRestoreWorkArray(dm,n*dim*dim*nc,PETSC_SCALAR,&sH);CHKERRQ(ierr);
    }
    if (D) {
      for (i = 0; i < n * dim * nc; i++) {D[i] = PetscRealPart(sD[i]);}
      ierr = DMGetWorkArray(dm,n*dim*nc,PETSC_SCALAR,&sD);CHKERRQ(ierr);
    }
    if (B) {
      for (i = 0; i < n * nc; i++) {B[i] = PetscRealPart(sB[i]);}
      ierr = DMGetWorkArray(dm,n*nc,PETSC_SCALAR,&sB);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFE_DA(DMField field, PetscInt nCells, const PetscInt *cells, PetscQuadrature points, PetscScalar *B, PetscScalar *D, PetscScalar *H)
{
  PetscInt       c, i, j, k, dim, cellsPer[3] = {0}, first[3] = {0};
  PetscReal      stepPer[3] = {0.};
  PetscReal      cellCoordRange[3][2] = {{-1.,2.},{-1.,2.},{-1.,2.}};
  PetscReal      *cellVals;
  DM             dm;
  DMDALocalInfo  info;
  PetscInt       cStart, cEnd;
  PetscInt       nq, nc;
  const PetscReal *q;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    *qs;
#else
  const PetscScalar *qs;
#endif
  DMField_DA     *dafield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dafield = (DMField_DA *) field->data;
  dm = field->dm;
  nc = field->numComponents;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  dim = info.dim;
  stepPer[0] = 1./ info.mx;
  stepPer[1] = 1./ info.my;
  stepPer[2] = 1./ info.mz;
  first[0] = info.gxs;
  first[1] = info.gys;
  first[2] = info.gzs;
  cellsPer[0] = info.gxm;
  cellsPer[1] = info.gym;
  cellsPer[2] = info.gzm;
  /* TODO: probably take components into account */
  ierr = PetscQuadratureGetData(points, NULL, NULL, &nq, &q, NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = DMGetWorkArray(dm,nq * dim,PETSC_SCALAR,&qs);CHKERRQ(ierr);
  for (i = 0; i < nq * dim; i++) qs[i] = q[i];
#else
  qs = q;
#endif
  ierr = DMDAGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,(1 << dim) * nc,PETSC_SCALAR,&cellVals);CHKERRQ(ierr);
  for (c = 0; c < nCells; c++) {
    PetscInt  cell = cells[c];
    PetscInt  rem  = cell;
    PetscInt  ijk[3] = {0};
    PetscReal eta0[3] = {0.};
    PetscReal *cB, *cD, *cH;

    cB = B ? &B[nc * nq * c] : NULL;
    cD = D ? &D[nc * nq * dim * c] : NULL;
    cH = H ? &H[nc * nq * dim * dim * c] : NULL;
    if (cell < cStart || cell >= cEnd) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Point %D not a cell [%D,%D), not implemented yet",cell,cStart,cEnd);
    for (i = 0; i < dim; i++) {
      ijk[i] = (rem % cellsPer[i]);
      rem /= cellsPer[i];
      eta0[i] = (ijk[i] + first[i]) * stepPer[i];
    }
    for (j = 0; j < (1 << dim); j++) {
      PetscReal eta[3];

      for (i = 0; i < nc; i++) {
        cellVals[j * nc + i] = 0.;
      }
      for (i = 0; i < dim; i++) {
        eta[i] = eta0[i] + ((j & (1 << i)) ? stepPer[i] : 0.);
      }
      for (k = 0; k < (1 << dim); k++) {
        PetscReal w = 1.;

        for (i = 0; i < dim; i++) {
          w *= (k & (1 << i)) ? eta[i] : (1. - eta[i]);
        }
        for (i = 0; i < nc; i++) {
          cellVals[j * nc + i] += w * dafield->cornerVals[k * nc + i];
        }
      }
    }
    MultilinearEvaluate(dim,cellCoordRange,nc,cellVals,nq,qs,cB,cD,cH);
  }
  ierr = DMRestoreWorkArray(dm,(1 << dim) * nc,PETSC_SCALAR,&cellVals);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = DMRestoreWorkArray(dm,nq * dim,PETSC_SCALAR,&qs);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFEReal_DA(DMField field, PetscInt numCells, const PetscInt *cells, PetscQuadrature points, PetscReal *B, PetscReal *D, PetscReal *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ierr = DMFieldEvaluateFE_DA(field,numCells,cells,points,B,D,H);CHKERRQ(ierr);
#else
  {
    DM          dm = field->dm;
    PetscInt    dim, i, nq;
    PetscScalar *sB = NULL, *sD = NULL, *sH = NULL;

    nc = field->numComponents;
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(points,NULL,NULL,&nq,NULL,NULL);CHKERRQ(ierr);
    if (B) {ierr = DMGetWorkArray(dm,nq*nc,PETSC_SCALAR,&sB);CHKERRQ(ierr);}
    if (D) {ierr = DMGetWorkArray(dm,nq*dim*nc,PETSC_SCALAR,&sD);CHKERRQ(ierr);}
    if (H) {ierr = DMGetWorkArray(dm,nq*dim*dim*nc,PETSC_SCALAR,&sH);CHKERRQ(ierr);}
    ierr = DMFieldEvaluateFE_DA(field,numCels,cells,points,sB,sD,sH);CHKERRQ(ierr);
    if (H) {
      for (i = 0; i < n * dim * dim * nc; i++) {H[i] = PetscRealPart(sH[i]);}
      ierr = DMRestoreWorkArray(dm,nq*dim*dim*nc,PETSC_SCALAR,&sH);CHKERRQ(ierr);
    }
    if (D) {
      for (i = 0; i < n * dim * nc; i++) {D[i] = PetscRealPart(sD[i]);}
      ierr = DMGetWorkArray(dm,nq*dim*nc,PETSC_SCALAR,&sD);CHKERRQ(ierr);
    }
    if (B) {
      for (i = 0; i < n * nc; i++) {B[i] = PetscRealPart(sB[i]);}
      ierr = DMGetWorkArray(dm,nq*nc,PETSC_SCALAR,&sB);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFV_DA(DMField field, PetscInt numCells, const PetscInt *cells, PetscScalar *B, PetscScalar *D, PetscScalar *H)
{
  PetscInt       c, i, dim, cellsPer[3] = {0}, first[3] = {0};
  PetscReal      stepPer[3] = {0.};
  DM             dm;
  DMDALocalInfo  info;
  PetscInt       cStart, cEnd;
  PetscInt       nc;
  PetscReal      *points;
  DMField_DA     *dafield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dafield = (DMField_DA *) field->data;
  dm = field->dm;
  nc = field->numComponents;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  dim = info.dim;
  stepPer[0] = 1./ info.mx;
  stepPer[1] = 1./ info.my;
  stepPer[2] = 1./ info.mz;
  first[0] = info.gxs;
  first[1] = info.gys;
  first[2] = info.gzs;
  cellsPer[0] = info.gxm;
  cellsPer[1] = info.gym;
  cellsPer[2] = info.gzm;
  ierr = DMDAGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,dim * numCells,PETSC_SCALAR,&points);CHKERRQ(ierr);
  for (c = 0; c < numCells; c++) {
    PetscInt  cell = cells[i];
    PetscInt  rem  = cell;
    PetscInt  ijk[3] = {0};

    if (cell < cStart || cell >= cEnd) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Point %D not a cell [%D,%D), not implemented yet",cell,cStart,cEnd);
    for (i = 0; i < dim; i++) {
      ijk[i] = (rem % cellsPer[i]);
      rem /= cellsPer[i];
      points[dim * c + i] = (ijk[i] + first[i] + 0.5) * stepPer[i];
    }
  }
  MultilinearEvaluate(dim,dafield->coordRange,nc,dafield->cornerVals,numCells,points,B,D,H);
  ierr = DMRestoreWorkArray(dm,dim * numCells,PETSC_SCALAR,&points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFVReal_DA(DMField field, PetscInt numCells, const PetscInt *cells, PetscReal *B, PetscReal *D, PetscReal *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ierr = DMFieldEvaluateFV_DA(field,numCells,cells,B,D,H);CHKERRQ(ierr);
#else
  {
    DM          dm = field->dm;
    PetscInt    dim, i, nq;
    PetscScalar *sB = NULL, *sD = NULL, *sH = NULL;

    nc = field->numComponents;
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    if (B) {ierr = DMGetWorkArray(dm,numCells*nc,PETSC_SCALAR,&sB);CHKERRQ(ierr);}
    if (D) {ierr = DMGetWorkArray(dm,numCells*dim*nc,PETSC_SCALAR,&sD);CHKERRQ(ierr);}
    if (H) {ierr = DMGetWorkArray(dm,numCells*dim*dim*nc,PETSC_SCALAR,&sH);CHKERRQ(ierr);}
    ierr = DMFieldEvaluateFV_DA(field,numCels,cells,sB,sD,sH);CHKERRQ(ierr);
    if (H) {
      for (i = 0; i < n * dim * dim * nc; i++) {H[i] = PetscRealPart(sH[i]);}
      ierr = DMRestoreWorkArray(dm,n*dim*dim*nc,PETSC_SCALAR,&sH);CHKERRQ(ierr);
    }
    if (D) {
      for (i = 0; i < n * dim * nc; i++) {D[i] = PetscRealPart(sD[i]);}
      ierr = DMGetWorkArray(dm,n*dim*nc,PETSC_SCALAR,&sD);CHKERRQ(ierr);
    }
    if (B) {
      for (i = 0; i < n * nc; i++) {B[i] = PetscRealPart(sB[i]);}
      ierr = DMGetWorkArray(dm,n*nc,PETSC_SCALAR,&sB);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldInitialize_DA(DMField field)
{
  DM             dm;
  Vec            coords = NULL;
  PetscInt       dim, i, j, k;
  DMField_DA     *dafield = field->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  field->ops->destroy        = DMFieldDestroy_DA;
  field->ops->evaluate       = DMFieldEvaluate_DA;
  field->ops->evaluateReal   = DMFieldEvaluateReal_DA;
  field->ops->evaluateFE     = DMFieldEvaluateFE_DA;
  field->ops->evaluateFEReal = DMFieldEvaluateFEReal_DA;
  field->ops->evaluateFV     = DMFieldEvaluateFV_DA;
  field->ops->evaluateFVReal = DMFieldEvaluateFVReal_DA;
  field->ops->view           = DMFieldView_DA;
  dm = field->dm;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dm->coordinates) coords = dm->coordinates;
  else if (dm->coordinatesLocal) coords = dm->coordinatesLocal;
  if (coords) {
    PetscInt          n;
    const PetscScalar *array;
    PetscReal         mins[3][2] = {{PETSC_MAX_REAL,PETSC_MAX_REAL}};

    ierr = VecGetLocalSize(coords,&n);CHKERRQ(ierr);
    n /= dim;
    ierr = VecGetArrayRead(coords,&array);CHKERRQ(ierr);
    for (i = 0, k = 0; i < n; i++) {
      for (j = 0; j < dim; j++, k++) {
        PetscReal val = PetscRealPart(array[k]);

        mins[j][0] = PetscMin(mins[j][0],val);
        mins[j][1] = PetscMin(mins[j][1],-val);
      }
    }
    ierr = VecRestoreArrayRead(coords,&array);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(mins,&(dafield->coordRange[0][0]),2*dim,MPIU_REAL,MPI_MIN,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    for (j = 0; j < dim; j++) {
      dafield->coordRange[j][1] = -dafield->coordRange[j][1];
    }
  } else {
    for (j = 0; j < dim; j++) {
      dafield->coordRange[j][0] = 0.;
      dafield->coordRange[j][1] = 1.;
    }
  }
  for (j = 0; j < dim; j++) {
    dafield->coordRange[j][1] -= dafield->coordRange[j][0];
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMFieldCreate_DA(DMField field)
{
  DMField_DA     *dafield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(field,&dafield);CHKERRQ(ierr);
  field->data = dafield;
  ierr = DMFieldInitialize_DA(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldCreateDA(DM dm, PetscInt numComponents, const PetscScalar *cornerValues,DMField *field)
{
  DMField        b;
  DMField_DA     *dafield;
  PetscInt       dim, nv, i;
  PetscScalar    *cv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMFieldCreate(dm,numComponents,DMFIELD_VERTEX,&b);CHKERRQ(ierr);
  ierr = DMFieldSetType(b,DMFIELDDA);CHKERRQ(ierr);
  dafield = (DMField_DA *) b->data;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  nv = (1 << dim) * numComponents;
  ierr = PetscMalloc1(nv,&cv);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) cv[i] = cornerValues[i];
  dafield->cornerVals = cv;
  *field = b;
  PetscFunctionReturn(0);
}
