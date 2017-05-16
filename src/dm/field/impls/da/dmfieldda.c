#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/
#include <petsc/private/dmimpl.h> /*I "petscdm.h" I*/

typedef struct _n_DMField_DA
{
  PetscScalar *cornerVals;
  PetscReal   coordRange[3][2];
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

static PetscErrorCode DMFieldEvaluate_DA(DMField field, Vec points, PetscScalar *B, PetscScalar *D, PetscScalar *H)
{
  DM             dm;
  DMField_DA     *dafield;
  PetscInt       dim;
  PetscInt       N, n, i, j, k, l, m, p, nc;
  const PetscScalar *array;
  PetscReal (*coordRange)[2];
  const PetscScalar *cv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm      = field->dm;
  nc      = field->numComponents;
  dafield = (DMField_DA *) field->data;
  cv      = dafield->cornerVals;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(points,&N);CHKERRQ(ierr);
  if (N % dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Point vector size %D not divisible by coordinate dimension %D\n",N,dim);
  n = N / dim;
  coordRange = &(dafield->coordRange[0]);
  ierr = VecGetArrayRead(points,&array);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    const PetscScalar *point = &array[dim * i];
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
      PetscScalar *out = &B[nc * dim * i];

      for (l = 0; l < (1 << dim); l++) {
        for (m = 0; m < nc * dim; m++) out[m] = 0.;
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

      for (l = 0; l < (1 << dim); l++) {
        for (m = 0; m < nc * dim * dim; m++) out[m] = 0.;
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
    ierr = DMFieldEvaluate_DA(field,poins,sB,sD,sH);CHKERRQ(ierr);
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
  field->ops->destroy      = DMFieldDestroy_DA;
  field->ops->evaluate     = DMFieldEvaluate_DA;
  field->ops->evaluateReal = DMFieldEvaluateReal_DA;
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
