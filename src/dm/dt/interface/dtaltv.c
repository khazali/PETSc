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
  PetscInt       N = alt->N;
  PetscErrorCode ierr;

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
    PetscInt  Njk = alt->Nk[j+k];
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
    PetscInt  Njk = alt->Nk[j+k];
    PetscInt  Nk = alt->Nk[k];
    PetscInt  JKj, i;
    PetscInt *subset   = alt->subset;
    PetscInt *subsetjk = alt->work;
    PetscInt *subsetj  = alt->perm;
    PetscInt *subsetk  = &alt->perm[j];

    ierr = PetscDTBinomial(j+k,j,&JKj);CHKERRQ(ierr);
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
  }
  PetscFunctionReturn(0);
}

/* L: V -> W [|W| by |V| array], L*: altW -> altV */
PetscErrorCode PetscDTAltVPullback(PetscDTAltV altv, PetscDTAltV altw, const PetscReal *L, PetscInt k, const PetscReal *w, PetscReal *Lstarw)
{
  PetscInt        N, M, Nk, Mk, Nf, i, j, l, p;
  PetscReal      *Lw, *Lwv;
  PetscInt       *subsetw, *subsetv;
  PetscInt       *work, *perm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  PetscValidPointer(altw, 2);
  N = altv->N;
  M = altw->N;
  if (k < 0 || k > N || k > M) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  Mk = altw->Nk[k];
  Nk = altv->Nk[k];
  Nf = altv->fac[k];
  subsetw = altw->subset;
  subsetv = altv->subset;
  work = altv->work;
  perm = altv->perm;
  ierr = PetscMalloc2(N * k, &Lw, k * k, &Lwv);CHKERRQ(ierr);
  for (i = 0; i < Nk; i++) Lstarw[i] = 0.;
  for (i = 0; i < Mk; i++) {
    ierr = PetscDTEnumSubset(M, k, i, subsetw);CHKERRQ(ierr);
    for (j = 0; j < Nk; j++) {
      ierr = PetscDTEnumSubset(N, k, j, subsetv);CHKERRQ(ierr);
      for (p = 0; p < Nf; p++) {
        PetscReal prod;
        PetscBool isOdd;

        ierr = PetscDTEnumPerm(k, p, work, perm, &isOdd);CHKERRQ(ierr);
        prod = isOdd ? -w[i] : w[i];
        for (l = 0; l < k; l++) {
          prod *= L[subsetw[perm[l]] * N + subsetv[l]];
        }
        Lstarw[j] += prod;
      }
    }
  }
  ierr = PetscFree2(Lw, Lwv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVPullbackMatrix(PetscDTAltV altv, PetscDTAltV altw, const PetscReal *L, PetscInt k, PetscReal *Lstar)
{
  PetscInt        N, M, Nk, Mk, Nf, i, j, l, p;
  PetscReal      *Lw, *Lwv;
  PetscInt       *subsetw, *subsetv;
  PetscInt       *work, *perm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  PetscValidPointer(altw, 2);
  N = altv->N;
  M = altw->N;
  if (k < 0 || k > N || k > M) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  Mk = altw->Nk[k];
  Nk = altv->Nk[k];
  Nf = altv->fac[k];
  subsetw = altw->subset;
  subsetv = altv->subset;
  work = altv->work;
  perm = altv->perm;
  ierr = PetscMalloc2(N * k, &Lw, k * k, &Lwv);CHKERRQ(ierr);
  for (i = 0; i < Nk * Mk; i++) Lstar[i] = 0.;
  for (i = 0; i < Mk; i++) {
    ierr = PetscDTEnumSubset(M, k, i, subsetw);CHKERRQ(ierr);
    for (j = 0; j < Nk; j++) {
      ierr = PetscDTEnumSubset(N, k, j, subsetv);CHKERRQ(ierr);
      for (p = 0; p < Nf; p++) {
        PetscReal prod;
        PetscBool isOdd;

        ierr = PetscDTEnumPerm(k, p, work, perm, &isOdd);CHKERRQ(ierr);
        prod = isOdd ? -1. : 1.;
        for (l = 0; l < k; l++) {
          prod *= L[subsetw[perm[l]] * N + subsetv[l]];
        }
        Lstar[j * Mk + i] += prod;
      }
    }
  }
  ierr = PetscFree2(Lw, Lwv);CHKERRQ(ierr);
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
  Nk = altv->Nk[k];
  Nkm = altv->Nk[k-1];
  subset = altv->subset;
  work = altv->work;
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
  Nk = altv->Nk[k];
  Nkm = altv->Nk[k-1];
  subset = altv->subset;
  work = altv->work;
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
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTAltVStar(PetscDTAltV altv, PetscInt k, const PetscReal *w, PetscReal *starw)
{
  PetscInt        N, Nk, i;
  PetscInt       *subset;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(altv, 1);
  N = altv->N;
  subset = altv->subset;
  if (k < 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  Nk = altv->Nk[k];
  for (i = 0; i < Nk; i++) {
    PetscBool sOdd;
    PetscInt  j, idx;

    ierr = PetscDTEnumSplit(N, k, i, subset, &sOdd);CHKERRQ(ierr);
    ierr = PetscDTSubsetIndex(N, k, subset, &idx);CHKERRQ(ierr);
    ierr = PetscDTSubsetIndex(N, N-k, &subset[k], &j);CHKERRQ(ierr);
    starw[j] = sOdd ? -w[idx] : w[idx];
  }
  PetscFunctionReturn(0);
}
