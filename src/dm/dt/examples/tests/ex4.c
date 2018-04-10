static char help[] = "Tests dual space symmetry.\n\n";

#include <petscfe.h>
#include <petscdmplex.h>

static PetscErrorCode CheckSymmetry(PetscInt dim, PetscInt order, PetscBool tensor)
{
  DM                dm;
  PetscDualSpace    sp;
  PetscInt          nFunc, i, closureSize, *closure = NULL, offset, depth;
  DMLabel           depthLabel;
  PetscBool         printed = PETSC_FALSE;
  PetscScalar       *vals, *valsCopy, *valsCopy2;
  const PetscInt    *numDofs;
  const PetscInt    **nnzs = NULL;
  const PetscInt    (***ijs)[2] = NULL;
  const PetscScalar ***symvals = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceCreate(PETSC_COMM_SELF,&sp);CHKERRQ(ierr);
  ierr = DMPlexCreateReferenceCell(PETSC_COMM_SELF,dim,tensor ? PETSC_FALSE : PETSC_TRUE,&dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(sp,PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(sp,dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(sp,order);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetContinuity(sp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(sp,tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(sp);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(sp);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(sp,&nFunc);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetSymmetries(sp,&nnzs,&ijs,&symvals);CHKERRQ(ierr);
  if (!nnzs && !ijs && !symvals) {
    ierr = PetscDualSpaceDestroy(&sp);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc3(nFunc*dim,&vals,nFunc*dim,&valsCopy,nFunc*dim,&valsCopy2);CHKERRQ(ierr);
  for (i = 0; i < nFunc; i++) {
    PetscQuadrature q;
    PetscInt        numPoints, Nc, j;
    const PetscReal *points;
    const PetscReal *weights;

    ierr = PetscDualSpaceGetFunctional(sp,i,&q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q,NULL,&Nc,&numPoints,&points,&weights);CHKERRQ(ierr);
    if (Nc != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only support scalar quadrature, not %D components\n",Nc);
    for (j = 0; j < dim; j++) vals[dim * i + j] = valsCopy2[dim * i + j] = (PetscScalar) points[j];
  }
  ierr = PetscDualSpaceGetNumDof(sp,&numDofs);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
  ierr = DMPlexGetTransitiveClosure(dm,0,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm,&depthLabel);CHKERRQ(ierr);
  for (i = 0, offset = 0; i < closureSize; i++, offset += numDofs[depth]) {
    PetscInt          point = closure[2 * i], coneSize, j;
    const PetscInt    (**pointijs)[2] = ijs ? ijs[i] : NULL;
    const PetscInt    *pointnnzs = nnzs ? nnzs[i] : NULL;
    const PetscScalar **pointVals = symvals ? symvals[i] : NULL;
    PetscBool         anyPrinted = PETSC_FALSE;

    ierr = DMLabelGetValue(depthLabel,point,&depth);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm,point,&coneSize);CHKERRQ(ierr);

    if (!pointnnzs && !pointijs && !pointVals) continue;
    for (j = -coneSize; j < coneSize; j++) {
      PetscInt          k, l;
      PetscInt          nnz = pointnnzs ? pointnnzs[j] : 0;
      const PetscInt    (*ij)[2] = pointijs ? pointijs[j] : NULL;
      const PetscScalar *symval = pointVals ? pointVals[j] : NULL;

      for (k = 0; k < numDofs[depth] * dim; k++) valsCopy[k] = 0.;
      for (k = 0; k < (nnz ? nnz : numDofs[depth]); k++) {
        PetscInt kLocal = ij ? ij[k][0] : k;
        PetscInt lLocal = ij ? ij[k][1] : k;

        for (l = 0; l < dim; l++) {
          valsCopy[kLocal * dim + l] += vals[(offset + lLocal) * dim + l] * (symval ? symval[k] : 1.);
        }
      }
      if (!printed && numDofs[depth] > 1) {
        Vec  vec;
        char name[256];

        anyPrinted = PETSC_TRUE;
        ierr = PetscSNPrintf(name,256,"%DD, %s, Order %D, Point %D Symmetry %D",dim,tensor ? "Tensor" : "Simplex", order, point,j);CHKERRQ(ierr);
        if (nnz && ij) {
          ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"IJs:\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
          ierr = PetscIntView(2 * nnz, (const PetscInt *) &(ij[0][0]), PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        }
        if (nnz && symvals) {
          ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Vals:\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
          ierr = PetscScalarView(nnz, vals, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        }
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,dim,numDofs[depth]*dim,valsCopy,&vec);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)vec,name);CHKERRQ(ierr);
        ierr = VecView(vec,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        ierr = VecDestroy(&vec);CHKERRQ(ierr);
      }
      for (k = 0; k < numDofs[depth] * dim; k++) valsCopy2[offset * dim + k] = 0.;
      for (k = 0; k < (nnz ? nnz : numDofs[depth]); k++) {
        PetscInt kLocal = ij ? ij[k][0] : k;
        PetscInt lLocal = ij ? ij[k][1] : k;

        for (l = 0; l < dim; l++) {
          valsCopy2[(offset + lLocal) * dim + l] += valsCopy[kLocal * dim + l] * (symval ? PetscConj(symval[k]) : 1.);
        }
      }
      for (k = 0; k < nFunc; k++) {
        for (l = 0; l < dim; l++) {
          PetscScalar diff = valsCopy2[dim * k + l] - vals[dim * k + l];
          if (PetscAbsScalar(diff) > PETSC_SMALL) SETERRQ8(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Symmetry failure: point %D, symmetry %D, order %D, functional %D, component %D: (%g - %g) = %g",point,j,order,k,l,(double) PetscRealPart(valsCopy2[dim * k + l]),(double) PetscRealPart(vals[dim * k + l]), (double) PetscRealPart(diff));
        }
      }
    }
    if (anyPrinted && !printed) printed = PETSC_TRUE;
  }
  ierr = DMPlexRestoreTransitiveClosure(dm,0,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  ierr = PetscFree3(vals,valsCopy,valsCopy2);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&sp);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       dim, order, tensor;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  for (tensor = 0; tensor < 2; tensor++) {
    for (dim = 1; dim <= 3; dim++) {
      if (dim == 1 && tensor) continue;
      for (order = 0; order <= (tensor ? 5 : 6); order++) {
        ierr = CheckSymmetry(dim,order,tensor ? PETSC_TRUE : PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
TEST*/
