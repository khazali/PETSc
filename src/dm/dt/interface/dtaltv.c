#include <petscdt.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dtimpl.h>

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
  PetscInt       N = alt->N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(alt, 1);
  if (k < 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  {
    PetscInt Nk, Nf;
    PetscInt *subset, *work, *perm;
    PetscInt i, j, l;
    PetscReal sum = 0.;

    ierr = PetscDTFactorialInt_Internal(k, &Nf);CHKERRQ(ierr);
    ierr = PetscDTBinomial(N, k, &Nk);CHKERRQ(ierr);
    ierr = PetscMalloc3(k, &subset, k, &work, k, &perm);CHKERRQ(ierr);
    for (i = 0; i < Nk; i++) {
      PetscReal subsum = 0.;

      ierr = PetscDTEnumSubset(N, k, i, subset);CHKERRQ(ierr);
      for (j = 0; j < Nf; j++) {
        PetscBool permOdd;
        PetscReal prod;

        ierr = PetscDTEnumPerm(k, j, work, perm, &permOdd);CHKERRQ(ierr);
        prod = permOdd ? -1. : 1.;
        for (l = 0; l < k; l++) {
          prod *= v[perm[l] * N + subset[l]];
        }
        subsum += prod;
      }
      sum += w[i] * subsum;
    }
    ierr = PetscFree3(subset, work, perm);CHKERRQ(ierr);
    *wv = sum;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVWedge(PetscDTAltV alt, PetscInt j, PetscInt k, const PetscReal *a, const PetscReal *b, PetscReal *awedgeb)
{
  PetscInt       N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(alt, 1);
  N = alt->N;
  if (j < 0 || k < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "negative form degree");
  if (j + k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wedge greater than dimension");
  {
    PetscInt  Njk;
    PetscInt  JKj;
    PetscInt *subset, *subsetjk, *subsetj, *subsetk;
    PetscInt  i;

    ierr = PetscDTBinomial(N, j+k, &Njk);CHKERRQ(ierr);
    ierr = PetscDTBinomial(j+k, j, &JKj);CHKERRQ(ierr);
    ierr = PetscMalloc4(j+k, &subset, j+k, &subsetjk, j, &subsetj, k, &subsetk);CHKERRQ(ierr);
    for (i = 0; i < Njk; i++) {
      PetscReal sum = 0.;
      PetscInt  l;

      ierr = PetscDTEnumSubset(N, j+k, i, subset);CHKERRQ(ierr);
      for (l = 0; l < JKj; l++) {
        PetscBool jkOdd;
        PetscInt  m, jInd, kInd;

        ierr = PetscDTEnumSplit(j+k, j, l, subsetjk, &jkOdd);CHKERRQ(ierr);
        for (m = 0; m < j; m++) {
          subsetj[m] = subset[subsetjk[m]];
        }
        for (m = 0; m < k; m++) {
          subsetk[m] = subset[subsetjk[j+m]];
        }
        ierr = PetscDTSubsetIndex(N, j, subsetj, &jInd);CHKERRQ(ierr);
        ierr = PetscDTSubsetIndex(N, k, subsetk, &kInd);CHKERRQ(ierr);
        sum += jkOdd ? -(a[jInd] * b[kInd]) : (a[jInd] * b[kInd]);
      }
      awedgeb[i] = sum;
    }
    ierr = PetscFree4(subset, subsetjk, subsetj, subsetk);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVWedgeMatrix(PetscDTAltV alt, PetscInt j, PetscInt k, const PetscReal *a, PetscReal *awedgeMat)
{
  PetscInt       N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(alt, 1);
  N = alt->N;
  if (j < 0 || k < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "negative form degree");
  if (j + k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wedge greater than dimension");
  {
    PetscInt  Njk;
    PetscInt  Nk;
    PetscInt  JKj, i;
    PetscInt *subset, *subsetjk, *subsetj, *subsetk;

    ierr = PetscDTBinomial(N,   k,   &Nk);CHKERRQ(ierr);
    ierr = PetscDTBinomial(N,   j+k, &Njk);CHKERRQ(ierr);
    ierr = PetscDTBinomial(j+k, j,   &JKj);CHKERRQ(ierr);
    ierr = PetscMalloc4(j+k, &subset, j+k, &subsetjk, j, &subsetj, k, &subsetk);CHKERRQ(ierr);
    for (i = 0; i < Njk * Nk; i++) awedgeMat[i] = 0.;
    for (i = 0; i < Njk; i++) {
      PetscInt  l;

      ierr = PetscDTEnumSubset(N, j+k, i, subset);CHKERRQ(ierr);
      for (l = 0; l < JKj; l++) {
        PetscBool jkOdd;
        PetscInt  m, jInd, kInd;

        ierr = PetscDTEnumSplit(j+k, j, l, subsetjk, &jkOdd);CHKERRQ(ierr);
        for (m = 0; m < j; m++) {
          subsetj[m] = subset[subsetjk[m]];
        }
        for (m = 0; m < k; m++) {
          subsetk[m] = subset[subsetjk[j+m]];
        }
        ierr = PetscDTSubsetIndex(N, j, subsetj, &jInd);CHKERRQ(ierr);
        ierr = PetscDTSubsetIndex(N, k, subsetk, &kInd);CHKERRQ(ierr);
        awedgeMat[i * Nk + kInd] += jkOdd ? - a[jInd] : a[jInd];
      }
    }
    ierr = PetscFree4(subset, subsetjk, subsetj, subsetk);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* L: V -> W [|W| by |V| array], L*: altW -> altV */
PetscErrorCode PetscDTAltVPullback(PetscDTAltV altv, PetscDTAltV altw, const PetscReal *L, PetscInt k, const PetscReal *w, PetscReal *Lstarw)
{
  PetscInt         N, M, Nk, Mk, Nf, i, j, l, p;
  PetscReal       *Lw, *Lwv;
  PetscInt        *subsetw, *subsetv;
  PetscInt        *work, *perm;
  PetscReal       *walloc = NULL;
  const PetscReal *ww = NULL;
  PetscBool        negative = PETSC_FALSE;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  PetscValidPointer(altw, 2);
  N = altv->N;
  M = altw->N;
  ierr = PetscDTBinomial(M, PetscAbsInt(k), &Mk);CHKERRQ(ierr);
  ierr = PetscDTBinomial(N, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
  ierr = PetscDTFactorialInt_Internal(PetscAbsInt(k), &Nf);CHKERRQ(ierr);
  if (k < 0) {
    negative = PETSC_TRUE;
    k = -k;
    ierr = PetscMalloc1(Mk, &walloc);CHKERRQ(ierr);
    ierr = PetscDTAltVStar(altw, M - k, 1, w, walloc);CHKERRQ(ierr);
    ww = walloc;
  } else {
    ww = w;
  }
  if (k > N || k > M) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  ierr = PetscMalloc6(k, &subsetw, k, &subsetv, k, &work, k, &perm, N * k, &Lw, k * k, &Lwv);CHKERRQ(ierr);
  for (i = 0; i < Nk; i++) Lstarw[i] = 0.;
  for (i = 0; i < Mk; i++) {
    ierr = PetscDTEnumSubset(M, k, i, subsetw);CHKERRQ(ierr);
    for (j = 0; j < Nk; j++) {
      ierr = PetscDTEnumSubset(N, k, j, subsetv);CHKERRQ(ierr);
      for (p = 0; p < Nf; p++) {
        PetscReal prod;
        PetscBool isOdd;

        ierr = PetscDTEnumPerm(k, p, work, perm, &isOdd);CHKERRQ(ierr);
        prod = isOdd ? -ww[i] : ww[i];
        for (l = 0; l < k; l++) {
          prod *= L[subsetw[perm[l]] * N + subsetv[l]];
        }
        Lstarw[j] += prod;
      }
    }
  }
  if (negative) {
    PetscReal *sLsw;

    ierr = PetscMalloc1(Nk, &sLsw);CHKERRQ(ierr);
    ierr = PetscDTAltVStar(altv, N - k, -1,  Lstarw, sLsw);CHKERRQ(ierr);
    for (i = 0; i < Nk; i++) Lstarw[i] = sLsw[i];
    ierr = PetscFree(sLsw);CHKERRQ(ierr);
  }
  ierr = PetscFree6(subsetw, subsetv, work, perm, Lw, Lwv);CHKERRQ(ierr);
  ierr = PetscFree(walloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVPullbackMatrix(PetscDTAltV altv, PetscDTAltV altw, const PetscReal *L, PetscInt k, PetscReal *Lstar)
{
  PetscInt        N, M, Nk, Mk, Nf, i, j, l, p;
  PetscReal      *Lw, *Lwv;
  PetscInt       *subsetw, *subsetv;
  PetscInt       *work, *perm;
  PetscBool       negative = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  PetscValidPointer(altw, 2);
  if (k < 0) {
    negative = PETSC_TRUE;
    k = -k;
  }
  N = altv->N;
  M = altw->N;
  if (k > N || k > M) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  ierr = PetscDTBinomial(M, PetscAbsInt(k), &Mk);CHKERRQ(ierr);
  ierr = PetscDTBinomial(N, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
  ierr = PetscDTFactorialInt_Internal(PetscAbsInt(k), &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc6(M, &subsetw, N, &subsetv, k, &work, k, &perm, N * k, &Lw, k * k, &Lwv);CHKERRQ(ierr);
  for (i = 0; i < Nk * Mk; i++) Lstar[i] = 0.;
  for (i = 0; i < Mk; i++) {
    PetscBool iOdd;
    PetscInt  iidx, jidx;

    ierr = PetscDTEnumSplit(M, k, i, subsetw, &iOdd);CHKERRQ(ierr);
    iidx = negative ? Mk - 1 - i : i;
    iOdd = negative ? iOdd ^ ((k * (M-k)) & 1) : PETSC_FALSE;
    for (j = 0; j < Nk; j++) {
      PetscBool jOdd;

      ierr = PetscDTEnumSplit(N, k, j, subsetv, &jOdd);CHKERRQ(ierr);
      jidx = negative ? Nk - 1 - j : j;
      jOdd = negative ? iOdd ^ jOdd ^ ((k * (N-k)) & 1) : PETSC_FALSE;
      for (p = 0; p < Nf; p++) {
        PetscReal prod;
        PetscBool isOdd;

        ierr = PetscDTEnumPerm(k, p, work, perm, &isOdd);CHKERRQ(ierr);
        isOdd ^= jOdd;
        prod = isOdd ? -1. : 1.;
        for (l = 0; l < k; l++) {
          prod *= L[subsetw[perm[l]] * N + subsetv[l]];
        }
        Lstar[jidx * Mk + iidx] += prod;
      }
    }
  }
  ierr = PetscFree6(subsetw, subsetv, work, perm, Lw, Lwv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVInterior(PetscDTAltV altv, PetscInt k, const PetscReal *w, const PetscReal *v, PetscReal *wIntv)
{
  PetscInt        N, i, Nk, Nkm;
  PetscInt       *subset, *work;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  N = altv->N;
  if (k <= 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  ierr = PetscDTBinomial(N, k,   &Nk);CHKERRQ(ierr);
  ierr = PetscDTBinomial(N, k-1, &Nkm);CHKERRQ(ierr);
  ierr = PetscMalloc2(k, &subset, k, &work);CHKERRQ(ierr);
  for (i = 0; i < Nkm; i++) wIntv[i] = 0.;
  for (i = 0; i < Nk; i++) {
    PetscInt  j, l, m;

    ierr = PetscDTEnumSubset(N, k, i, subset);CHKERRQ(ierr);
    for (j = 0; j < k; j++) {
      PetscInt  idx;
      PetscBool flip = (j & 1);

      for (l = 0, m = 0; l < k; l++) {
        if (l != j) work[m++] = subset[l];
      }
      ierr = PetscDTSubsetIndex(N, k - 1, work, &idx);CHKERRQ(ierr);
      wIntv[idx] += flip ? -(w[i] * v[subset[j]]) :  (w[i] * v[subset[j]]);
    }
  }
  ierr = PetscFree2(subset, work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVInteriorMatrix(PetscDTAltV altv, PetscInt k, const PetscReal *v, PetscReal *intvMat)
{
  PetscInt        N, i, Nk, Nkm;
  PetscInt       *subset, *work;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  N = altv->N;
  if (k <= 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  ierr = PetscDTBinomial(N, k,   &Nk);CHKERRQ(ierr);
  ierr = PetscDTBinomial(N, k-1, &Nkm);CHKERRQ(ierr);
  ierr = PetscMalloc2(k, &subset, k, &work);CHKERRQ(ierr);
  for (i = 0; i < Nk * Nkm; i++) intvMat[i] = 0.;
  for (i = 0; i < Nk; i++) {
    PetscInt  j, l, m;

    ierr = PetscDTEnumSubset(N, k, i, subset);CHKERRQ(ierr);
    for (j = 0; j < k; j++) {
      PetscInt  idx;
      PetscBool flip = (j & 1);

      for (l = 0, m = 0; l < k; l++) {
        if (l != j) work[m++] = subset[l];
      }
      ierr = PetscDTSubsetIndex(N, k - 1, work, &idx);CHKERRQ(ierr);
      intvMat[idx * Nk + i] += flip ? -v[subset[j]] :  v[subset[j]];
    }
  }
  ierr = PetscFree2(subset, work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVStar(PetscDTAltV altv, PetscInt k, PetscInt pow, const PetscReal *w, PetscReal *starw)
{
  PetscInt        N, Nk, i;
  PetscInt       *subset;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  N = altv->N;
  if (k < 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  ierr = PetscDTBinomial(N, k, &Nk);CHKERRQ(ierr);
  ierr = PetscMalloc1(N, &subset);CHKERRQ(ierr);
  pow = pow % 4;
  pow = (pow + 4) % 4; /* make non-negative */
  /* pow is now 0, 1, 2, 3 */
  if (pow % 2) {
    PetscInt l = (pow == 1) ? k : N - k;
    for (i = 0; i < Nk; i++) {
      PetscBool sOdd;
      PetscInt  j, idx;

      ierr = PetscDTEnumSplit(N, l, i, subset, &sOdd);CHKERRQ(ierr);
      ierr = PetscDTSubsetIndex(N, l, subset, &idx);CHKERRQ(ierr);
      ierr = PetscDTSubsetIndex(N, N-l, &subset[l], &j);CHKERRQ(ierr);
      starw[j] = sOdd ? -w[idx] : w[idx];
    }
  } else {
    for (i = 0; i < Nk; i++) starw[i] = w[i];
  }
  /* star^2 = -1^(k * (N - k)) */
  if (pow > 1 && (k * (N - k)) % 2) {
    for (i = 0; i < Nk; i++) starw[i] = -starw[i];
  }
  ierr = PetscFree(subset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
