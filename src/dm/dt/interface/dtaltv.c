#include <petscdt.h>
#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

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
    ierr = PetscDTBinomial_Internal(N, i, &(av->Nk[i]));CHKERRQ(ierr);
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

  PetscFunctionBeginHot;
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
      PetscBool setOdd;

      PetscDTEnumSubsetWithSign(N, k, Nk, i, subset, &setOdd);
      for (j = 0; j < Nf; j++) {
        PetscBool permOdd;

        PetscDTEnumPermWithSign(k, j, work, perm, &permOdd);
        PetscReal prod = permOdd ? -1. : 1.;
        for (l = 0; l < k; l++) {
          prod *= v[perm[l] * N + subset[l]];
        }
        subsum += prod;
      }
      /* use the sign of the subset to keep consistent with determinant calculation */
      //sum += setOdd ? -(w[i] * subsum) : (w[i] * subsum);
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

  PetscFunctionBeginHot;
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

    ierr = PetscDTBinomial_Internal(j+k,j,&JKj);CHKERRQ(ierr);
    for (i = 0; i < Njk; i++) {
      PetscReal sum = 0.;
      PetscInt  l;
      PetscBool setOdd;

      PetscDTEnumSubsetWithSign(N, j+k, Njk, i, subset, &setOdd);
      for (l = 0; l < Nj; l++) {
        PetscBool jkOdd, jOdd, kOdd;
        PetscInt  m, cj, ck, jInd, kInd;
        PetscReal jMult, kMult;

        PetscDTEnumSubsetWithSign(j+k, j, JKj, l, subsetjk, &jkOdd);
        for (m = 0, cj = 0, ck = 0; m < j+k; m++) {
          if (subsetjk[cj] == m) {
            subsetj[cj++] = subset[m];
          } else {
            subsetk[ck++] = subset[m];
          }
        }
        PetscDTSubsetIndexWithSign(N, j, Nj, subsetj, &jInd, &jOdd);
        PetscDTSubsetIndexWithSign(N, k, Nk, subsetk, &kInd, &kOdd);
        jMult = jOdd ? -a[jInd] : a[jInd];
        kMult = kOdd ? -b[kInd] : b[kInd];
        sum += jkOdd ? -(jMult * kMult) : (jMult * kMult);
      }
      awedgeb[i] = setOdd ? -sum : sum;
    }
  }
  PetscFunctionReturn(0);
}

