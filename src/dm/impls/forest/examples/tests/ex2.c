static char help[] = "Create a mesh, refine and coarsen simultaneously, and transfer a field\n\n";

#include <petscds.h>
#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscoptions.h>

static PetscErrorCode AddIdentityLabel(DM dm)
{
  PetscInt       pStart,pEnd,p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLabel(dm, "identity");CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {ierr = DMSetLabelValue(dm, "identity", p, p);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAdaptivityLabel(DM forest,DMLabel *adaptLabel)
{
  DMLabel        identLabel;
  PetscInt       cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelCreate("adapt",adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelSetDefaultValue(*adaptLabel,DM_ADAPT_COARSEN);CHKERRQ(ierr);
  ierr = DMGetLabel(forest,"identity",&identLabel);CHKERRQ(ierr);
  ierr = DMForestGetCellChart(forest,&cStart,&cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; c++) {
    PetscInt basePoint;

    ierr = DMLabelGetValue(identLabel,c,&basePoint);CHKERRQ(ierr);
    if (!basePoint) {ierr = DMLabelSetValue(*adaptLabel,c,DM_ADAPT_REFINE);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode LinearFunction(PetscInt dim,PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscFunctionBeginUser;
  u[0] = (x[0] * 2.0 + 1.) + (x[1] * 20.0 + 10.) + ((dim == 3) ? (x[2] * 200.0 + 100.) : 0.);
  PetscFunctionReturn(0);
}

static PetscErrorCode MultiaffineFunction(PetscInt dim,PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscFunctionBeginUser;
  u[0] = (x[0] * 1.0 + 2.0) * (x[1] * 3.0 - 4.0) * ((dim == 3) ? (x[2] * 5.0 + 6.0) : 1.);
  PetscFunctionReturn(0);
}

static PetscErrorCode CoordsFunction(PetscInt dim,PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscInt f;

  PetscFunctionBeginUser;
  for (f=0;f<Nf;f++) u[f] = x[f];
  PetscFunctionReturn(0);
}

typedef struct _bc_func_ctx
{
  PetscErrorCode (*func) (PetscInt,PetscReal,const PetscReal [], PetscInt, PetscScalar [], void *);
  PetscInt dim;
  PetscInt Nf;
  void *ctx;
}
bc_func_ctx;

static PetscErrorCode bc_func_fv (PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  bc_func_ctx    *bcCtx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bcCtx = (bc_func_ctx *) ctx;
  ierr = (bcCtx->func)(bcCtx->dim,time,c,bcCtx->Nf,xG,bcCtx->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IdentifyBadPoints (DM dm, Vec vec, PetscReal tol)
{
  DM             dmplex;
  PetscInt       p, pStart, pEnd, maxDof;
  Vec            vecLocal;
  DMLabel        depthLabel;
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(dm, &vecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, vec, INSERT_VALUES, vecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, vec, INSERT_VALUES, vecLocal);CHKERRQ(ierr);
  ierr = DMConvert(dm ,DMPLEX, &dmplex);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dmplex, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dmplex, &depthLabel);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dmplex, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    PetscInt     s, c, cSize, parent, childID, numChildren;
    PetscInt     cl, closureSize, *closure = NULL;
    PetscScalar *values = NULL;
    PetscBool    bad = PETSC_FALSE;

    ierr = VecGetValuesSection(vecLocal, section, p, &values);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(section, p, &cSize);CHKERRQ(ierr);
    for (c = 0; c < cSize; c++) {
      PetscReal absDiff = PetscAbsScalar(values[c]);CHKERRQ(ierr);
      if (absDiff > tol) {bad = PETSC_TRUE; break;}
    }
    if (!bad) continue;
    ierr = PetscPrintf(PETSC_COMM_SELF, "Bad point %D\n", p);CHKERRQ(ierr);
    ierr = DMLabelGetValue(depthLabel, p, &s);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "  Depth %D\n", s);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(dmplex, p, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize; cl++) {
      PetscInt cp = closure[2 * cl];
      ierr = DMPlexGetTreeParent(dmplex, cp, &parent, &childID);CHKERRQ(ierr);
      if (parent != cp) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  Closure point %D (%D) child of %D (ID %D)\n", cl, cp, parent, childID);CHKERRQ(ierr);
      }
      ierr = DMPlexGetTreeChildren(dmplex, cp, &numChildren, NULL);CHKERRQ(ierr);
      if (numChildren) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  Closure point %D (%D) is parent\n", cl, cp);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dmplex, p, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (c = 0; c < cSize; c++) {
      PetscReal absDiff = PetscAbsScalar(values[c]);CHKERRQ(ierr);
      if (absDiff > tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  Bad dof %D\n", c);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMDestroy(&dmplex);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             base, preForest, postForest;
  PetscInt       dim = 2, Nf = 1;
  PetscInt       preCount, postCount;
  Vec            preVec, postVecTransfer, postVecExact;
  PetscErrorCode (*funcs[1]) (PetscInt,PetscReal,const PetscReal [],PetscInt,PetscScalar [], void *) = {MultiaffineFunction};
  void           *ctxs[1] = {NULL};
  const PetscInt cells[] = {3, 3, 3};
  PetscReal      diff, tol = PETSC_SMALL;
  PetscBool      linear = PETSC_FALSE;
  PetscBool      coords = PETSC_FALSE;
  PetscBool      useFV = PETSC_FALSE;
  PetscBool      conv = PETSC_FALSE;
  PetscBool      transfer_from_base = PETSC_FALSE;
  PetscBool      use_bcs = PETSC_TRUE;
  PetscDS        ds;
  bc_func_ctx    bcCtx;
  DMLabel        adaptLabel;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMForestTransferVec() Test Options", "DMFOREST");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The dimension (2 or 3)", "ex2.c", dim, &dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-linear","Transfer a simple linear function", "ex2.c", linear, &linear, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-coords","Transfer a simple coordinate function", "ex2.c", coords, &coords, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_fv","Use a finite volume approximation", "ex2.c", useFV, &useFV, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_convert","Test conversion to DMPLEX",NULL,conv,&conv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-transfer_from_base","Transfer a vector from base DM to DMForest", "ex2.c", transfer_from_base, &transfer_from_base, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_bcs","Use dirichlet boundary conditions", "ex2.c", transfer_from_base, &transfer_from_base, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (linear) {
    funcs[0] = LinearFunction;
  }
  if (coords) {
    funcs[0] = CoordsFunction;
    Nf = dim;
  }

  bcCtx.func = funcs[0];
  bcCtx.dim  = dim;
  bcCtx.Nf   = Nf;
  bcCtx.ctx  = NULL;

  /* the base mesh */
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, cells, NULL, NULL, NULL, PETSC_TRUE, &base);CHKERRQ(ierr);
  if (useFV) {
    PetscFV      fv;
    PetscLimiter limiter;
    DM           baseFV;

    ierr = DMPlexConstructGhostCells(base,NULL,NULL,&baseFV);CHKERRQ(ierr);
    ierr = DMDestroy(&base);CHKERRQ(ierr);
    base = baseFV;
    ierr = PetscFVCreate(comm, &fv);CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fv,dim);CHKERRQ(ierr);
    ierr = PetscFVSetType(fv,PETSCFVLEASTSQUARES);CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fv,Nf);CHKERRQ(ierr);
    ierr = PetscLimiterCreate(comm,&limiter);CHKERRQ(ierr);
    ierr = PetscLimiterSetType(limiter,PETSCLIMITERNONE);CHKERRQ(ierr);
    ierr = PetscFVSetLimiter(fv,limiter);CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&limiter);CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(fv);CHKERRQ(ierr);
    ierr = DMSetField(base,0,(PetscObject)fv);CHKERRQ(ierr);
    ierr = PetscFVDestroy(&fv);CHKERRQ(ierr);
  } else {
    PetscFE fe;
    ierr = PetscFECreateDefault(comm,dim,Nf,PETSC_FALSE,NULL,PETSC_DEFAULT,&fe);CHKERRQ(ierr);
    ierr = DMSetField(base,0,(PetscObject)fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  }
  if (use_bcs) {
    PetscDS  prob;
    PetscInt ids[]   = {1, 2, 3, 4, 5, 6};

    ierr = DMGetDS(base,&prob);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob,DM_BC_ESSENTIAL, "bc", "marker", 0, 0, NULL, useFV ? (void(*)(void)) bc_func_fv : (void(*)(void)) funcs[0], 2 * dim, ids, useFV ? (void *) &bcCtx : NULL);CHKERRQ(ierr);
  }
  ierr = AddIdentityLabel(base);CHKERRQ(ierr);
  ierr = DMViewFromOptions(base,NULL,"-dm_base_view");CHKERRQ(ierr);

  /* the pre adaptivity forest */
  ierr = DMCreate(comm,&preForest);CHKERRQ(ierr);
  ierr = DMSetType(preForest,(dim == 2) ? DMP4EST : DMP8EST);CHKERRQ(ierr);
  ierr = DMGetDS(base,&ds);CHKERRQ(ierr);
  ierr = DMSetDS(preForest,ds);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(preForest,base);CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(preForest,1);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(preForest,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(preForest);CHKERRQ(ierr);
  ierr = DMSetUp(preForest);CHKERRQ(ierr);
  ierr = DMViewFromOptions(preForest,NULL,"-dm_pre_view");CHKERRQ(ierr);

  /* the pre adaptivity field */
  ierr = DMCreateGlobalVector(preForest,&preVec);CHKERRQ(ierr);
  ierr = DMProjectFunction(preForest,0.,funcs,ctxs,INSERT_VALUES,preVec);CHKERRQ(ierr);
  ierr = VecViewFromOptions(preVec,NULL,"-vec_pre_view");CHKERRQ(ierr);

  /* communicate between base and pre adaptivity forest */
  if (transfer_from_base) {
    DM  preForestPlex;
    Mat inject = NULL;
    Vec baseVec, baseVecMapped;

    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented");
    ierr = DMCreateGlobalVector(base,&baseVec);CHKERRQ(ierr);
    ierr = DMProjectFunction(base,0.,funcs,ctxs,INSERT_VALUES,baseVec);CHKERRQ(ierr);
    ierr = VecViewFromOptions(baseVec,NULL,"-vec_base_view");CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(preForest,&baseVecMapped);CHKERRQ(ierr);
    ierr = DMConvert(preForest,DMPLEX,&preForestPlex);CHKERRQ(ierr);
    /* none of these alternatives work */
    //ierr = DMSetCoarseDM(preForestPlex,base);CHKERRQ(ierr);
    //ierr = DMPlexSetRegularRefinement(preForestPlex,PETSC_TRUE);CHKERRQ(ierr);
    //ierr = DMCreateInterpolation(base,preForestPlex,&inject,NULL);CHKERRQ(ierr);

    //ierr = DMCreateInjection(base,preForestPlex,&inject);CHKERRQ(ierr);

    //ierr = DMCreateInjection(preForestPlex,base,&inject);CHKERRQ(ierr);

    //ierr = DMCreateRestriction(base,preForestPlex,&inject);CHKERRQ(ierr);
    (void)(inject);
    ierr = MatInterpolate(inject,baseVec,baseVecMapped);CHKERRQ(ierr);
    ierr = VecViewFromOptions(baseVecMapped,NULL,"-vec_basemap_view");CHKERRQ(ierr);

    /* compare */
    ierr = VecAXPY(baseVecMapped,-1.,preVec);CHKERRQ(ierr);
    ierr = VecViewFromOptions(baseVecMapped,NULL,"-vec_diff_view");CHKERRQ(ierr);
    ierr = VecNorm(baseVecMapped,NORM_2,&diff);CHKERRQ(ierr);

    /* output */
    if (diff < tol) {
      ierr = PetscPrintf(comm,"Transfer vec from DMPLEX to DMFOREST passes.\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(comm,"Transfer vec from DMPLEX to DMFOREST fails with error %g and tolerance %g\n",diff,tol);CHKERRQ(ierr);
    }

    ierr = DMDestroy(&preForestPlex);CHKERRQ(ierr);
    ierr = VecDestroy(&baseVec);CHKERRQ(ierr);
    ierr = VecDestroy(&baseVecMapped);CHKERRQ(ierr);
  }

  ierr = PetscObjectGetReference((PetscObject)preForest,&preCount);CHKERRQ(ierr);

  /* adapt */
  ierr = CreateAdaptivityLabel(preForest,&adaptLabel);CHKERRQ(ierr);
  ierr = DMForestTemplate(preForest,comm,&postForest);CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(postForest,0);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(postForest,0);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityLabel(postForest,adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  ierr = DMSetUp(postForest);CHKERRQ(ierr);
  ierr = DMViewFromOptions(postForest,NULL,"-dm_post_view");CHKERRQ(ierr);

  /* transfer */
  ierr = DMCreateGlobalVector(postForest,&postVecTransfer);CHKERRQ(ierr);
  ierr = DMForestTransferVec(preForest,preVec,postForest,postVecTransfer,PETSC_TRUE,0.0);CHKERRQ(ierr);
  ierr = VecViewFromOptions(postVecTransfer,NULL,"-vec_post_transfer_view");CHKERRQ(ierr);

  /* the exact post adaptivity field */
  ierr = DMCreateGlobalVector(postForest,&postVecExact);CHKERRQ(ierr);
  ierr = DMProjectFunction(postForest,0.,funcs,ctxs,INSERT_VALUES,postVecExact);CHKERRQ(ierr);
  ierr = VecViewFromOptions(postVecExact,NULL,"-vec_post_exact_view");CHKERRQ(ierr);

  /* compare */
  ierr = VecAXPY(postVecTransfer,-1.,postVecExact);CHKERRQ(ierr);
  ierr = VecViewFromOptions(postVecTransfer,NULL,"-vec_diff_view");CHKERRQ(ierr);
  ierr = VecNorm(postVecTransfer,NORM_2,&diff);CHKERRQ(ierr);

  /* output */
  if (diff < tol) {
    ierr = PetscPrintf(comm,"DMForestTransferVec() passes.\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(comm,"DMForestTransferVec() fails with error %g and tolerance %g\n",diff,tol);CHKERRQ(ierr);
    ierr = IdentifyBadPoints(postForest, postVecTransfer, tol);CHKERRQ(ierr);
  }

  /* disconnect preForest from postForest */
  ierr = DMForestSetAdaptivityForest(postForest,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetReference((PetscObject)preForest,&postCount);CHKERRQ(ierr);
  if (postCount != preCount) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Adaptation not memory neutral: reference count increase from %d to %d\n",preCount,postCount);

  if (conv) {
    DM dmConv;

    ierr = DMConvert(postForest,DMPLEX,&dmConv);CHKERRQ(ierr);
    ierr = DMPlexCheckCellShape(dmConv,PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dmConv,NULL,"-dm_conv_view");CHKERRQ(ierr);
    ierr = DMDestroy(&dmConv);CHKERRQ(ierr);
  }

  /* cleanup */
  ierr = VecDestroy(&postVecExact);CHKERRQ(ierr);
  ierr = VecDestroy(&postVecTransfer);CHKERRQ(ierr);
  ierr = DMDestroy(&postForest);CHKERRQ(ierr);
  ierr = VecDestroy(&preVec);CHKERRQ(ierr);
  ierr = DMDestroy(&preForest);CHKERRQ(ierr);
  ierr = DMDestroy(&base);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       output_file: output/ex2_2d.out
       suffix: p4est_2d
       args: -petscspace_type tensor -petscspace_degree 2 -dim 2
       nsize: 3
       requires: p4est

     test:
       output_file: output/ex2_2d.out
       suffix: p4est_2d_deg4
       args: -petscspace_type tensor -petscspace_degree 4 -dim 2
       requires: p4est

     test:
       output_file: output/ex2_2d.out
       suffix: p4est_2d_deg8
       args: -petscspace_type tensor -petscspace_degree 8 -dim 2
       requires: p4est

     test:
       output_file: output/ex2_2d_fv.out
       suffix: p4est_2d_fv
       args: -use_fv -linear -dim 2 -dm_forest_partition_overlap 1
       nsize: 3
       requires: p4est

     test:
       TODO: broken (codimension adjacency)
       output_file: output/ex2_2d_fv.out
       suffix: p4est_2d_fv_adjcodim
       args: -use_fv -linear -dim 2 -dm_forest_partition_overlap 1 -dm_forest_adjacency_codimension 1
       nsize: 2
       requires: p4est

     test:
       TODO: broken (dimension adjacency)
       output_file: output/ex2_2d_fv.out
       suffix: p4est_2d_fv_adjdim
       args: -use_fv -linear -dim 2 -dm_forest_partition_overlap 1 -dm_forest_adjacency_dimension 1
       nsize: 2
       requires: p4est

     test:
       TODO: broken (zero cells on one process?)
       output_file: output/ex2_2d_fv.out
       suffix: p4est_2d_fv_zerocells
       args: -use_fv -linear -dim 2 -dm_forest_partition_overlap 1
       nsize: 10
       requires: p4est

     test:
       output_file: output/ex2_3d.out
       suffix: p4est_3d
       args: -petscspace_type tensor -petscspace_degree 1 -dim 3
       nsize: 3
       requires: p4est

     test:
       output_file: output/ex2_3d.out
       suffix: p4est_3d_deg3
       args: -petscspace_type tensor -petscspace_degree 3 -dim 3
       nsize: 3
       requires: p4est

     test:
       output_file: output/ex2_2d.out
       suffix: p4est_2d_deg2_coords
       args: -petscspace_type tensor -petscspace_degree 2 -dim 2 -coords
       nsize: 3
       requires: p4est

     test:
       output_file: output/ex2_3d.out
       suffix: p4est_3d_deg2_coords
       args: -petscspace_type tensor -petscspace_degree 2 -dim 3 -coords
       nsize: 3
       requires: p4est

     test:
       output_file: output/ex2_3d_fv.out
       suffix: p4est_3d_fv
       args: -use_fv -linear -dim 3 -dm_forest_partition_overlap 1
       nsize: 3
       requires: p4est

     test:
       suffix: p4est_3d_nans
       args: -dim 3 -dm_forest_partition_overlap 1 -test_convert -dm_conv_view ::ascii_info_detail -petscspace_type tensor -petscspace_degree 1
       nsize: 2

TEST*/
