/*
  Common tools for constructing discretizations
*/
#if !defined(__PETSCDT_H)
#define __PETSCDT_H

#include <petscsys.h>

/*S
  PetscQuadrature - Quadrature rule for integration.

  Level: developer

.seealso:  PetscQuadratureCreate(), PetscQuadratureDestroy()
S*/
typedef struct _p_PetscQuadrature *PetscQuadrature;

PETSC_EXTERN PetscErrorCode PetscQuadratureCreate(MPI_Comm, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureDuplicate(PetscQuadrature, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetOrder(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetOrder(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetNumComponents(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetNumComponents(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetData(PetscQuadrature, PetscInt*, PetscInt*, PetscInt*, const PetscReal *[], const PetscReal *[]);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetData(PetscQuadrature, PetscInt, PetscInt, PetscInt, const PetscReal [], const PetscReal []);
PETSC_EXTERN PetscErrorCode PetscQuadratureView(PetscQuadrature, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscQuadratureDestroy(PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscQuadratureExpandComposite(PetscQuadrature, PetscInt, const PetscReal[], const PetscReal[], PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscDTLegendreEval(PetscInt,const PetscReal*,PetscInt,const PetscInt*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussQuadrature(PetscInt,PetscReal,PetscReal,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTReconstructPoly(PetscInt,PetscInt,const PetscReal*,PetscInt,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussTensorQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);
PETSC_EXTERN PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);

PETSC_EXTERN PetscErrorCode PetscDTTanhSinhTensorQuadrature(PetscInt, PetscInt, PetscReal, PetscReal, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrate(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrateMPFR(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);

/*S
  PetscDTAltV - Alternating algebraic k-form calculations

  Level: developer
S*/
typedef struct _n_PetscDTAltV *PetscDTAltV;

PETSC_EXTERN PetscErrorCode PetscDTAltVCreate(PetscInt, PetscDTAltV *);
PETSC_EXTERN PetscErrorCode PetscDTAltVDestroy(PetscDTAltV *);
PETSC_EXTERN PetscErrorCode PetscDTAltVGetN(PetscDTAltV, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDTAltVGetSize(PetscDTAltV, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDTAltVApply(PetscDTAltV, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVWedge(PetscDTAltV, PetscInt, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVWedgeMatrix(PetscDTAltV, PetscInt, PetscInt, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVPullback(PetscDTAltV, PetscDTAltV, const PetscReal *, PetscInt, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVPullbackMatrix(PetscDTAltV, PetscDTAltV, const PetscReal *, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVInterior(PetscDTAltV, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVInteriorMatrix(PetscDTAltV, PetscInt, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVStar(PetscDTAltV, PetscInt, const PetscReal *, PetscReal *);

PETSC_STATIC_INLINE PetscErrorCode PetscDTBinomial(PetscInt n, PetscInt k, PetscInt *binomial)
{
  PetscReal f = 1.0;
  PetscReal g = 1.0;
  PetscErrorCode ierr;
  PetscInt  i;

  PetscFunctionBegin;
  if (n < 0 || k < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Binomal arguments (%D %D) must be non-negative\n", n, k);
  if (2 * k < n) {
    ierr = PetscDTBinomial(n, n - k, binomial);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (i = k + 1; i <= n; i++) f *= i;
  for (i = 1; i <= n - k; i++) g *= i;
  *binomial = (PetscInt) (f / g);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void PetscDTEnumPermWithSign(PetscInt n, PetscInt k, PetscInt *work, PetscInt *perm, PetscBool *isOdd)
{
  PetscBool odd = PETSC_FALSE;
  PetscInt  i;
  PetscInt *w = &work[n - 2];

  PetscFunctionBeginHot;
  i = 2;
  for (i = 2; i <= n; i++) {
    *(w--) = k % i;
    k /= i;
  }
  for (i = 0; i < n; i++) perm[i] = i;
  for (i = 0; i < n - 1; i++) {
    PetscInt s = work[i];
    PetscInt swap = perm[i];

    perm[i] = perm[i + s];
    perm[i + s] = swap;
    odd ^= (!!s);
  }
  *isOdd = odd;
  PetscFunctionReturnVoid();
}

PETSC_STATIC_INLINE void PetscDTEnumSubset(PetscInt n, PetscInt k, PetscInt Nk, PetscInt j, PetscInt *subset)
{
  PetscInt i, l;

  PetscFunctionBeginHot;
  for (i = 0, l = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (j < Nminuskminus) {
      subset[l++] = i;
      Nk = Nminuskminus;
    } else {
      j -= Nminuskminus;
      Nk = Nminusk;
    }
  }
  PetscFunctionReturnVoid();
}

PETSC_STATIC_INLINE void PetscDTEnumSplitWithSign(PetscInt n, PetscInt k, PetscInt Nk, PetscInt j, PetscInt *subset, PetscBool *isOdd)
{
  PetscInt i, l, m, *subcomp;
  PetscBool odd;

  PetscFunctionBeginHot;
  odd = PETSC_FALSE;
  subcomp = &subset[k];
  for (i = 0, l = 0, m = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (j < Nminuskminus) {
      subset[l++] = i;
      Nk = Nminuskminus;
    } else {
      subcomp[m++] = i;
      j -= Nminuskminus;
      odd ^= ((k - l) & 1);
      Nk = Nminusk;
    }
  }
  for (; i < n; i++) {
    subcomp[m++] = i;
  }
  *isOdd = odd;
  PetscFunctionReturnVoid();
}

PETSC_STATIC_INLINE void PetscDTSubsetIndex(PetscInt n, PetscInt k, PetscInt Nk, const PetscInt *subset, PetscInt *index)
{
  PetscInt  j = 0;
  PetscInt  i, l;

  PetscFunctionBeginHot;
  for (i = 0, l = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (subset[l] == i) {
      l++;
      Nk = Nminuskminus;
    } else {
      j += Nminuskminus;
      Nk = Nminusk;
    }
  }
  *index = j;
  PetscFunctionReturnVoid();
}

#endif
