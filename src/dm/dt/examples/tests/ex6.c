static char help[] = "Tests alternating forms.\n\n";

#include <petscviewer.h>
#include <petscdt.h>

int main(int argc, char **argv)
{
  PetscInt       i, numTests = 5, n[5] = {0, 1, 2, 3, 4};
  PetscBool      verbose = PETSC_FALSE;
  PetscRandom    rand;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for alternating form test","none");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-N", "Up to 5 vector space dimensions to test","ex6.c",n,&numTests,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-verbose", "Verbose test output","ex6.c",verbose,&verbose,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand, -1., 1.);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  if (!numTests) numTests = 5;
  viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
  for (i = 0; i < numTests; i++) {
    PetscDTAltV    altv;
    PetscInt       k, N = n[i], m;

    ierr = PetscViewerASCIIPrintf(viewer, "N =  %D:\n", N);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);

    ierr = PetscDTAltVCreate(N, &altv);CHKERRQ(ierr);
    ierr = PetscDTAltVGetN(altv, &m);CHKERRQ(ierr);
    if (m != N) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "AltV n set/get mismatch: %D != %D\n", N, m);
    if (verbose) {
      PetscInt *work, *perm;
      PetscInt fac = 1;

      ierr = PetscMalloc2(N, &perm, N, &work);CHKERRQ(ierr);

      for (k = 1; k <= N; k++) fac *= k;
      ierr = PetscViewerASCIIPrintf(viewer, "Permutations of %D:\n", N);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      for (k = 0; k < fac; k++) {
        PetscBool isOdd;
        PetscInt  j;

        PetscDTEnumPermWithSign(N, k, work, perm, &isOdd);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%D:", k);CHKERRQ(ierr);
        for (j = 0; j < N; j++) {
          ierr = PetscPrintf(PETSC_COMM_WORLD, " %D", perm[j]);CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_WORLD, ", %s\n", isOdd ? "odd" : "even");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscFree2(perm,work);CHKERRQ(ierr);
    }
    for (k = 0; k <= N; k++) {
      PetscInt   j, Nk;
      PetscReal *w, *v, wv;
      PetscInt  *subset;

      ierr = PetscDTAltVGetSize(altv, k, &Nk);CHKERRQ(ierr);
      if (verbose) {ierr = PetscViewerASCIIPrintf(viewer, "(%D choose %D): %D\n", N, k, Nk);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);

      /* Test subset enumeration */
      ierr = PetscMalloc1(k, &subset);CHKERRQ(ierr);
      for (j = 0; j < Nk; j++) {
        PetscBool isOdd, isOddCheck;
        PetscInt  jCheck;
        PetscDTEnumSubsetWithSign(N, k, Nk, j, subset, &isOdd);
        if (verbose) {
          PetscInt l;

          ierr = PetscViewerASCIIPrintf(viewer, "subset %D:", j);CHKERRQ(ierr);
          for (l = 0; l < k; l++) {
            ierr = PetscPrintf(PETSC_COMM_WORLD, " %D", subset[l]);CHKERRQ(ierr);
          }
          ierr = PetscPrintf(PETSC_COMM_WORLD, ", %s\n", isOdd ? "odd" : "even");CHKERRQ(ierr);
        }
        PetscDTSubsetIndexWithSign(N, k, Nk, subset, &jCheck, &isOddCheck);
        if (jCheck != j) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "jCheck (%D) != j (%D)", jCheck, j);
        if (isOdd != isOddCheck) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "sign check");
      }
      ierr = PetscFree(subset);CHKERRQ(ierr);

      /* Make a random k form */
      ierr = PetscMalloc1(Nk, &w);CHKERRQ(ierr);
      for (j = 0; j < Nk; j++) {ierr = PetscRandomGetValueReal(rand, &w[j]);CHKERRQ(ierr);}
      /* Make a set of random vectors */
      ierr = PetscMalloc1(N*k, &v);CHKERRQ(ierr);
      for (j = 0; j < N*k; j++) {ierr = PetscRandomGetValueReal(rand, &v[j]);CHKERRQ(ierr);}

      ierr = PetscDTAltVApply(altv, k, w, v, &wv);CHKERRQ(ierr);

      if (verbose) {
        ierr = PetscViewerASCIIPrintf(viewer, "w:\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        if (Nk) {ierr = PetscRealView(Nk, w, viewer);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "v:\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        if (N*k > 0) {ierr = PetscRealView(N*k, v, viewer);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "w(v): %g\n", (double) wv);CHKERRQ(ierr);
      }

      /* sanity checks */
      if (k == 1) {
        PetscInt  l;
        PetscReal wvcheck = 0.;
        PetscReal diff;

        for (l = 0; l < N; l++) wvcheck += w[l] * v[l];
        diff = PetscSqrtReal(PetscSqr(wvcheck - wv));CHKERRQ(ierr);
        if (diff >= PETSC_SMALL * (PetscAbsReal(wv) + PetscAbsReal(wvcheck))) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "1-form / dot product equivalence: wvcheck (%g) != wv (%g)", (double) wvcheck, (double) wv);
      }
      if (k == N && N < 4) {
        PetscReal det, wvcheck, diff;

        switch (k) {
        case 0:
          det = 1.;
          break;
        case 1:
          det = v[0];
          break;
        case 2:
          det = v[0] * v[3] - v[1] * v[0];
          break;
        case 3:
          det = v[0] * (v[4] * v[8] - v[5] * v[7]) +
                v[1] * (v[5] * v[6] - v[3] * v[8]) +
                v[2] * (v[3] * v[7] - v[4] * v[6]);
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "invalid k");
        }
        wvcheck = det * w[0];
        diff = PetscSqrtReal(PetscSqr(wvcheck - wv));CHKERRQ(ierr);
        if (diff >= PETSC_SMALL * (PetscAbsReal(wv) + PetscAbsReal(wvcheck))) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "n-form / determinant equivalence: wvcheck (%g) != wv (%g)", (double) wvcheck, (double) wv);
      }
      ierr = PetscFree(v);CHKERRQ(ierr);
      ierr = PetscFree(w);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscDTAltVDestroy(&altv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    args: -verbose
TEST*/
