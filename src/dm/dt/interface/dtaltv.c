#include <petscdt.h>
#include <petsc/private/petscimpl.h>

struct _n_PetscDTAltV {
  PetscInt   N;
  PetscInt  *Nk;
  PetscInt  *fac;
  PetscInt  *subset;
  PetscInt  *subsetjk;
  PetscInt  *perm;
  PetscInt  *work;
  PetscInt  *Nint;
  PetscInt **intInd;
};

PetscErrorCode PetscDTAltVCreate(PetscInt N, PetscDTAltV *alt)
{
  PetscDTAltV    av;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&av);CHKERRQ(ierr);
  av->N = N;
  ierr = PetscMalloc5(N + 1, &(av->Nk), N + 1, &(av->fac), N, &(av->subset), N, &(av->work), N, &(av->perm));CHKERRQ(ierr);
  for (i = 0; i <= N; i++) {
    ierr = PetscDTBinomial(N, i, &(av->Nk[i]));CHKERRQ(ierr);
  }
  av->fac[0] = 1;
  for (i = 1; i <= N; i++) {
    av->fac[i] = i * av->fac[i-1];
  }
  *alt = av;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVDestroy(PetscDTAltV *alt)
{
  PetscDTAltV    av = *alt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree5(av->Nk, av->fac, av->subset, av->work, av->perm);CHKERRQ(ierr);
  ierr = PetscFree(*alt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVGetN(PetscDTAltV alt, PetscInt *N)
{
  PetscFunctionBegin;
  PetscValidPointer(alt, 1);
  PetscValidIntPointer(N, 2);
  *N = alt->N;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVGetSize(PetscDTAltV alt, PetscInt k, PetscInt *Nk)
{
  PetscFunctionBegin;
  PetscValidPointer(alt, 1);
  PetscValidIntPointer(Nk, 3);
  if (k < 0 || k > alt->N) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "k (%D) must be in [0, %D]\n", k, alt->N);
  *Nk = alt->Nk[k];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVApply(PetscDTAltV alt, PetscInt k, const PetscReal *w, const PetscReal *v, PetscReal *wv)
{
  PetscInt N = alt->N;

  PetscFunctionBegin;
  PetscValidPointer(alt, 1);
  if (k < 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  {
    PetscInt Nk = alt->Nk[k]; /* number of subsets of size k */
    PetscInt *subset = alt->subset;
    PetscInt Nf = alt->fac[k];
    PetscInt i, j, l;
    PetscInt *work = alt->work;
    PetscInt *perm = alt->perm;
    PetscReal sum = 0.;

    for (i = 0; i < Nk; i++) {
      PetscReal subsum = 0.;

      PetscDTEnumSubset(N, k, Nk, i, subset);
      for (j = 0; j < Nf; j++) {
        PetscBool permOdd;

        PetscDTEnumPermWithSign(k, j, work, perm, &permOdd);
        PetscReal prod = permOdd ? -1. : 1.;
        for (l = 0; l < k; l++) {
          prod *= v[perm[l] * N + subset[l]];
        }
        subsum += prod;
      }
      sum += w[i] * subsum;
    }
    *wv = sum;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVWedge(PetscDTAltV alt, PetscInt j, PetscInt k, const PetscReal *a, const PetscReal *b, PetscReal *awedgeb)
{
  PetscErrorCode ierr;
  PetscInt N = alt->N;

  PetscFunctionBegin;
  if (j < 0 || k < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "negative form degree");
  if (j + k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wedge greater than dimension");
  {
    PetscInt  Njk = alt->Nk[j+k];
    PetscInt  Nj = alt->Nk[j];
    PetscInt  Nk = alt->Nk[k];
    PetscInt  JKj;
    PetscInt *subset   = alt->subset;
    PetscInt *subsetjk = alt->work;
    PetscInt *subsetj  = alt->perm;
    PetscInt *subsetk  = &alt->perm[j];
    PetscInt  i;

    ierr = PetscDTBinomial(j+k,j,&JKj);CHKERRQ(ierr);
    for (i = 0; i < Njk; i++) {
      PetscReal sum = 0.;
      PetscInt  l;

      PetscDTEnumSubset(N, j+k, Njk, i, subset);
      for (l = 0; l < JKj; l++) {
        PetscBool jkOdd;
        PetscInt  m, jInd, kInd;

        PetscDTEnumSplitWithSign(j+k, j, JKj, l, subsetjk, &jkOdd);
        for (m = 0; m < j; m++) {
          subsetj[m] = subset[subsetjk[m]];
        }
        for (m = 0; m < k; m++) {
          subsetk[m] = subset[subsetjk[j+m]];
        }
        PetscDTSubsetIndex(N, j, Nj, subsetj, &jInd);
        PetscDTSubsetIndex(N, k, Nk, subsetk, &kInd);
        sum += jkOdd ? -(a[jInd] * b[kInd]) : (a[jInd] * b[kInd]);
      }
      awedgeb[i] = sum;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVWedgeMatrix(PetscDTAltV alt, PetscInt j, PetscInt k, const PetscReal *a, PetscReal *awedgeMat)
{
  PetscErrorCode ierr;
  PetscInt N = alt->N;

  PetscFunctionBegin;
  if (j < 0 || k < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "negative form degree");
  if (j + k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wedge greater than dimension");
  {
    PetscInt  Njk = alt->Nk[j+k];
    PetscInt  Nj = alt->Nk[j];
    PetscInt  Nk = alt->Nk[k];
    PetscInt  JKj;
    PetscInt *subset   = alt->subset;
    PetscInt *subsetjk = alt->work;
    PetscInt *subsetj  = alt->perm;
    PetscInt *subsetk  = &alt->perm[j];
    PetscInt  i;

    ierr = PetscDTBinomial(j+k,j,&JKj);CHKERRQ(ierr);
    for (i = 0; i < Njk * Nk; i++) awedgeMat[i] = 0.;
    for (i = 0; i < Njk; i++) {
      PetscInt  l;

      PetscDTEnumSubset(N, j+k, Njk, i, subset);
      for (l = 0; l < JKj; l++) {
        PetscBool jkOdd;
        PetscInt  m, jInd, kInd;

        PetscDTEnumSplitWithSign(j+k, j, JKj, l, subsetjk, &jkOdd);
        for (m = 0; m < j; m++) {
          subsetj[m] = subset[subsetjk[m]];
        }
        for (m = 0; m < k; m++) {
          subsetk[m] = subset[subsetjk[j+m]];
        }
        PetscDTSubsetIndex(N, j, Nj, subsetj, &jInd);
        PetscDTSubsetIndex(N, k, Nk, subsetk, &kInd);
        awedgeMat[i * Nk + kInd] += jkOdd ? - a[jInd] : a[jInd];
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTAltVPullback_Internal(PetscDTAltV altv, PetscInt N, PetscInt M, const PetscReal *L, PetscInt k, PetscInt Mk, const PetscReal *w, PetscReal *Lstarminus, PetscReal *Lstarw2, PetscReal *Lstarw)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  if (k == 1) { /* L^T * w */
    PetscInt i, j;

    for (i = 0; i < N; i++) Lstarw[i] = 0.;
    for (j = 0; j < M; j++) {
      for (i = 0; i < N; i++) {
        Lstarw[i] += L[j * N + i] * w[j];
      }
    }
  } else {
    PetscInt Mminuskminus, Mminusk, i;

    Mminuskminus = (Mk * k) / M;
    Mminusk = Mk - Mminuskminus;
    for (i = 0; i < altv->Nk[k-1]; i++) Lstarminus[i] = 0.;
    if (Mminuskminus) {ierr = PetscDTAltVPullback_Internal(altv, N, M-1, &L[N], k-1, Mminuskminus, w, &Lstarminus[altv->Nk[k-1]], Lstarw2, Lstarminus);CHKERRQ(ierr);}
    ierr = PetscDTAltVWedge(altv, 1, k-1, L, Lstarminus, Lstarw2);CHKERRQ(ierr);
    for (i = 0; i < altv->Nk[k]; i++) Lstarw[i] += Lstarw2[i];
    if (Mminusk) {ierr = PetscDTAltVPullback_Internal(altv, N, M-1, &L[N], k, Mminusk, &w[Mminuskminus], Lstarminus, Lstarw2, Lstarw);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/* L: V -> W [|W| by |V| array], L*: altW -> altV */
PetscErrorCode PetscDTAltVPullback(PetscDTAltV altv, PetscDTAltV altw, const PetscReal *L, PetscInt k, const PetscReal *w, PetscReal *Lstarw)
{
  PetscInt       N, M;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  PetscValidPointer(altw, 2);
  N = altv->N;
  M = altw->N;
  if (k < 0 || k > N || k > M) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  if (k == 0) {
    Lstarw[0] = w[0];
  } else {
    PetscReal *Lstarminus, *Lstarw2;
    PetscInt   Mk = altw->Nk[k];
    PetscInt   Nk = altv->Nk[k];
    PetscInt   tempSize, tempMax;
    PetscInt   i;

    tempSize = 0;
    tempMax  = 0;
    for (i = 0; i < k; i++) {
      tempSize += altv->Nk[i];
      tempMax   = PetscMax(tempMax,altv->Nk[i+1]);
    }
    ierr = PetscMalloc2(tempSize, &Lstarminus, tempMax, &Lstarw2);CHKERRQ(ierr);
    for (i = 0; i < Nk; i++) Lstarw[i] = 0.;
    ierr = PetscDTAltVPullback_Internal(altv, N, M, L, k, Mk, w, Lstarminus, Lstarw2, Lstarw);CHKERRQ(ierr);
    ierr = PetscFree2(Lstarminus, Lstarw2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTAltVPullbackMatrix_Internal(PetscDTAltV altv, PetscInt N, PetscInt M, const PetscReal *L, PetscInt k, PetscInt Mk, PetscReal *Lstarminus, PetscReal *Lstar2, PetscReal *Lstar, PetscInt rowSize)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  if (k == 1) { /* L^T */
    PetscInt i, j;

    for (i = 0; i < N; i++) {
      for (j = 0; j < M; j++) {
        Lstar[i * M + j] += L[j * N + i];
      }
    }
  } else {
    PetscInt Mminuskminus, Mminusk, i, j, l, matSize;
    PetscInt Nkminus = altv->Nk[k-1];

    Mminuskminus = (Mk * k) / M;
    Mminusk = Mk - Mminuskminus;
    matSize = Nkminus * Mminuskminus;
    for (i = 0; i < matSize; i++) Lstarminus[i] = 0.;
    if (Mminuskminus) {ierr = PetscDTAltVPullbackMatrix_Internal(altv, N, M-1, &L[N], k-1, Mminuskminus, &Lstarminus[matSize], Lstar2, Lstarminus, Mminuskminus);CHKERRQ(ierr);}
    ierr = PetscDTAltVWedgeMatrix(altv, 1, k-1, L, Lstar2);CHKERRQ(ierr);
    for (i = 0; i < altv->Nk[k]; i++) {
      for (j = 0; j < Mminuskminus; j++) {
        PetscReal sum = 0.;
        for (l = 0; l < Nkminus; l++) {
          sum += Lstar2[i * Nkminus + l] * Lstarminus[l * Mminuskminus + j];
        }
        Lstar[i * rowSize + j] += sum;
      }
    }
    if (Mminusk) {ierr = PetscDTAltVPullbackMatrix_Internal(altv, N, M-1, &L[N], k, Mminusk, Lstarminus, Lstar2, &Lstar[Mminuskminus], rowSize);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVPullbackMatrix(PetscDTAltV altv, PetscDTAltV altw, const PetscReal *L, PetscInt k, PetscReal *Lstar)
{
  PetscInt       N, M;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  PetscValidPointer(altw, 2);
  N = altv->N;
  M = altw->N;
  if (k < 0 || k > N || k > M) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  if (k == 0) {
    Lstar[0] = 1.;
  } else {
    PetscReal *Lstarminus, *Lstar2;
    PetscInt   Mk = altw->Nk[k];
    PetscInt   Nk = altv->Nk[k];
    PetscInt   tempSize, tempMax;
    PetscInt   i;

    tempSize = 0;
    tempMax  = 0;
    for (i = 0; i < k; i++) {
      tempSize += altv->Nk[i];
      tempMax   = PetscMax(tempMax,altv->Nk[i]*altv->Nk[i+1]);
    }
    ierr = PetscMalloc2(tempSize * altw->Nk[M / 2], &Lstarminus, tempMax, &Lstar2);CHKERRQ(ierr);
    for (i = 0; i < Nk * Mk; i++) Lstar[i] = 0.;
    ierr = PetscDTAltVPullbackMatrix_Internal(altv, N, M, L, k, Mk, Lstarminus, Lstar2, Lstar, Mk);CHKERRQ(ierr);
    ierr = PetscFree2(Lstarminus, Lstar2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVInterior(PetscDTAltV altv, PetscInt k, const PetscReal *w, const PetscReal *v, PetscReal *wIntv)
{
  PetscInt       N, i, Nk, Nkm;
  PetscInt       *subset, *work;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  N = altv->N;
  if (k <= 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  Nk = altv->Nk[k];
  Nkm = altv->Nk[k-1];
  subset = altv->subset;
  work = altv->work;
  for (i = 0; i < Nkm; i++) wIntv[i] = 0.;
  for (i = 0; i < Nk; i++) {
    PetscInt j, l, m;

    PetscDTEnumSubset(N, k, Nk, i, subset);
    for (j = 0; j < k; j++) {
      PetscInt idx;

      for (l = 0, m = 0; l < k; l++) {
        if (l != j) work[m++] = subset[l];
      }
      PetscDTSubsetIndex(N, k - 1, Nkm, work, &idx);
      wIntv[idx] += (j & 1) ? -(w[i] * v[subset[j]]) :  (w[i] * v[subset[j]]);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVInteriorMatrix(PetscDTAltV altv, PetscInt k, const PetscReal *v, PetscReal *intvMat)
{
  PetscInt       N, i, Nk, Nkm;
  PetscInt       *subset, *work;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  N = altv->N;
  if (k <= 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  Nk = altv->Nk[k];
  Nkm = altv->Nk[k-1];
  subset = altv->subset;
  work = altv->work;
  for (i = 0; i < Nk * Nkm; i++) intvMat[i] = 0.;
  for (i = 0; i < Nk; i++) {
    PetscInt j, l, m;

    PetscDTEnumSubset(N, k, Nk, i, subset);
    for (j = 0; j < k; j++) {
      PetscInt idx;

      for (l = 0, m = 0; l < k; l++) {
        if (l != j) work[m++] = subset[l];
      }
      PetscDTSubsetIndex(N, k - 1, Nkm, work, &idx);
      intvMat[idx * Nk + i] += (j & 1) ? -v[subset[j]] :  v[subset[j]];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVStar(PetscDTAltV altv, PetscInt k, const PetscReal *w, PetscReal *starw)
{
  PetscInt N, Nk, i;
  PetscInt *subset;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  N = altv->N;
  subset = altv->subset;
  if (k < 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  Nk = altv->Nk[k];
  for (i = 0; i < Nk; i++) {
    PetscBool isOdd;

    PetscDTEnumSplitWithSign(N, k, Nk, i, subset, &isOdd);
    starw[Nk-1-i] = isOdd ? -w[i] : w[i];
  }
  PetscFunctionReturn(0);
}
