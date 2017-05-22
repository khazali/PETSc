static char help[] = "Demonstration of creating and viewing DMFields objects.\n\n";

#include <petscdmfield.h>
#include <petscdmplex.h>
#include <petscdmda.h>

static PetscErrorCode ViewResults(PetscViewer viewer, PetscInt N, PetscInt dim, PetscScalar *B, PetscScalar *D, PetscScalar *H, PetscReal *rB, PetscReal *rD, PetscReal *rH)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"B:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(N,B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"D:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(N*dim,D,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"H:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(N*dim*dim,H,viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"rB:\n");CHKERRQ(ierr);
  ierr = PetscRealView(N,rB,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rD:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(N*dim,rD,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rH:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(N*dim*dim,rH,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEvaluate(DMField field, PetscInt n, PetscRandom rand)
{
  DM             dm;
  PetscInt       dim, i, nc;
  PetscScalar    *B, *D, *H;
  PetscReal      *rB, *rD, *rH;
  Vec            points;
  PetscScalar    *array;
  PetscViewer    viewer;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)field);
  ierr = DMFieldGetNumComponents(field,&nc);CHKERRQ(ierr);
  ierr = DMFieldGetDM(field,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)field),n * dim,PETSC_DETERMINE,&points);CHKERRQ(ierr);
  ierr = VecSetBlockSize(points,dim);CHKERRQ(ierr);
  ierr = VecGetArray(points,&array);CHKERRQ(ierr);
  for (i = 0; i < n * dim; i++) {ierr = PetscRandomGetValue(rand,&array[i]);CHKERRQ(ierr);}
  ierr = VecRestoreArray(points,&array);CHKERRQ(ierr);
  ierr = PetscMalloc6(n*nc,&B,n*nc,&rB,n*nc*dim,&D,n*nc*dim,&rD,n*nc*dim*dim,&H,n*nc*dim*dim,&rH);CHKERRQ(ierr);
  ierr = DMFieldEvaluate(field,points,PETSC_SCALAR,B,D,H);CHKERRQ(ierr);
  ierr = DMFieldEvaluate(field,points,PETSC_REAL,rB,rD,rH);CHKERRQ(ierr);
  viewer = PETSC_VIEWER_STDOUT_(comm);

  ierr = PetscObjectSetName((PetscObject)points,"Test Points");CHKERRQ(ierr);
  ierr = VecView(points,viewer);CHKERRQ(ierr);
  ierr = ViewResults(viewer,n*nc,dim,B,D,H,rB,rD,rH);CHKERRQ(ierr);

  ierr = PetscFree6(B,rB,D,rD,H,rH);CHKERRQ(ierr);
  ierr = VecDestroy(&points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEvaluateFE(DMField field, PetscInt n, PetscInt cStart, PetscInt cEnd, PetscQuadrature quad, PetscRandom rand)
{
  DM             dm;
  PetscInt       dim, i, nc, nq;
  PetscInt       N;
  PetscScalar    *B, *D, *H;
  PetscReal      *rB, *rD, *rH;
  PetscInt       *cells;
  PetscViewer    viewer;
  PetscBool      compact = PETSC_FALSE;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)field);
  ierr = PetscOptionsBegin(comm, "", "DMField TestEvaluateFE Options", "DM");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_fe_compact","Test DMFieldEvaluateFECompact()", "DM", compact, &compact, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = DMFieldGetNumComponents(field,&nc);CHKERRQ(ierr);
  ierr = DMFieldGetDM(field,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,(PetscScalar) cStart, (PetscScalar) cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&cells);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    PetscReal rc;

    ierr = PetscRandomGetValueReal(rand,&rc);CHKERRQ(ierr);
    cells[i] = PetscFloorReal(rc);
  }
  ierr = PetscQuadratureGetData(quad,NULL,NULL,&nq,NULL,NULL);CHKERRQ(ierr);
  N    = n * nq * nc;
  ierr = PetscMalloc6(N,&B,N,&rB,N*dim,&D,N*dim,&rD,N*dim*dim,&H,N*dim*dim,&rH);CHKERRQ(ierr);
  if (!compact) {
    ierr = DMFieldEvaluateFE(field,n,cells,quad,PETSC_SCALAR,B,D,H);CHKERRQ(ierr);
    ierr = DMFieldEvaluateFE(field,n,cells,quad,PETSC_REAL,rB,rD,rH);CHKERRQ(ierr);
  } else {
    PetscBool isConstant, isAffine, isQuadratic;

    ierr = DMFieldGetFEInvariance(field,n,cells,&isConstant,&isAffine,&isQuadratic);CHKERRQ(ierr);
    /* compact, in the presence of invariance, fills the output array as though there is one point per cell */
    ierr = DMFieldEvaluateFECompact(field,n,cells,quad,PETSC_SCALAR,isConstant,B,isAffine,D,isQuadratic,H);CHKERRQ(ierr);
    ierr = DMFieldEvaluateFECompact(field,n,cells,quad,PETSC_REAL,isConstant,rB,isAffine,rD,isQuadratic,rH);CHKERRQ(ierr);
    /* expand the compacted entries: should give the same result as DMFieldEvaluateFE() */
    for (i = n - 1; i >= 0; i--) {
      PetscInt j;
      for (j = nq - 1; j >= 0; j--) {
        if (isConstant) {
          ierr = PetscMemmove(&B[(i * nq + j)* nc],&B[i * nc],nc * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemmove(&rB[(i * nq + j)* nc],&rB[i * nc],nc * sizeof(PetscReal));CHKERRQ(ierr);
        }
        if (isAffine) {
          ierr = PetscMemmove(&D[(i * nq + j)* nc * dim],&D[i * nc * dim],nc * dim * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemmove(&rD[(i * nq + j)* nc * dim],&rD[i * nc * dim],nc * dim * sizeof(PetscReal));CHKERRQ(ierr);
        }
        if (isQuadratic) {
          ierr = PetscMemmove(&H[(i * nq + j)* nc * dim * dim],&H[i * nc * dim * dim],nc * dim * dim * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemmove(&rH[(i * nq + j)* nc * dim * dim],&rH[i * nc * dim * dim],nc * dim * dim * sizeof(PetscReal));CHKERRQ(ierr);
        }
      }
    }
  }
  viewer = PETSC_VIEWER_STDOUT_(comm);

  ierr = PetscObjectSetName((PetscObject)quad,"Test quadrature");CHKERRQ(ierr);
  ierr = PetscQuadratureView(quad,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Test Cells:\n");CHKERRQ(ierr);
  ierr = PetscIntView(n,cells,viewer);CHKERRQ(ierr);
  ierr = ViewResults(viewer,N,dim,B,D,H,rB,rD,rH);CHKERRQ(ierr);

  ierr = PetscFree6(B,rB,D,rD,H,rH);CHKERRQ(ierr);
  ierr = PetscFree(cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEvaluateFV(DMField field, PetscInt n, PetscInt cStart, PetscInt cEnd, PetscRandom rand)
{
  DM             dm;
  PetscInt       dim, i, nc;
  PetscInt       N;
  PetscScalar    *B, *D, *H;
  PetscReal      *rB, *rD, *rH;
  PetscInt       *cells;
  PetscViewer    viewer;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)field);
  ierr = DMFieldGetNumComponents(field,&nc);CHKERRQ(ierr);
  ierr = DMFieldGetDM(field,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,(PetscScalar) cStart, (PetscScalar) cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&cells);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    PetscReal rc;

    ierr = PetscRandomGetValueReal(rand,&rc);CHKERRQ(ierr);
    cells[i] = PetscFloorReal(rc);
  }
  N    = n * nc;
  ierr = PetscMalloc6(N,&B,N,&rB,N*dim,&D,N*dim,&rD,N*dim*dim,&H,N*dim*dim,&rH);CHKERRQ(ierr);
  ierr = DMFieldEvaluateFV(field,n,cells,PETSC_SCALAR,B,D,H);CHKERRQ(ierr);
  ierr = DMFieldEvaluateFV(field,n,cells,PETSC_REAL,rB,rD,rH);CHKERRQ(ierr);
  viewer = PETSC_VIEWER_STDOUT_(comm);

  ierr = PetscViewerASCIIPrintf(viewer,"Test Cells:\n");CHKERRQ(ierr);
  ierr = PetscIntView(n,cells,viewer);CHKERRQ(ierr);
  ierr = ViewResults(viewer,N,dim,B,D,H,rB,rD,rH);CHKERRQ(ierr);

  ierr = PetscFree6(B,rB,D,rD,H,rH);CHKERRQ(ierr);
  ierr = PetscFree(cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode radiusSquared(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscInt       i;
  PetscReal      r2 = 0.;

  PetscFunctionBegin;
  for (i = 0; i < dim; i++) {r2 += PetscSqr(x[i]);}
  for (i = 0; i < Nf; i++) {
    u[i] = (i + 1) * r2;
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM              dm = NULL;
  MPI_Comm        comm;
  char            type[256] = DMPLEX;
  PetscBool       isda, isplex;
  PetscInt        dim = 2;
  DMField         field = NULL;
  PetscInt        nc = 1;
  PetscInt        cStart = -1, cEnd = -1;
  PetscRandom     rand;
  PetscQuadrature quad = NULL;
  PetscInt        pointsPerEdge = 2;
  PetscInt        numPoint = 0, numFE = 0, numFV = 0;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMField Tutorial Options", "DM");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_type","DM implementation on which to define field","ex1.c",DMList,type,type,256,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","DM intrinsic dimension", "ex1.c", dim, &dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_components","Number of components in field", "ex1.c", nc, &nc, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_quad_points","Number of quadrature points per dimension", "ex1.c", pointsPerEdge, &pointsPerEdge, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_point_tests", "Number of test points for DMFieldEvaluate()", "ex1.c", numPoint, &numPoint, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_fe_tests", "Number of test cells for DMFieldEvaluateFE()", "ex1.c", numFE, &numFE, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_fv_tests", "Number of test cells for DMFieldEvaluateFV()", "ex1.c", numFV, &numFV, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (dim > 3) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"This examples works for dim <= 3, not %D",dim);
  ierr = PetscStrncmp(type,DMPLEX,256,&isplex);CHKERRQ(ierr);
  ierr = PetscStrncmp(type,DMDA,256,&isda);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  if (isplex) {
    PetscBool simplex = PETSC_TRUE;
    PetscInt  overlap = 0;
    Vec       fieldvec;
    PetscFE   fe;

    ierr = PetscOptionsBegin(comm, "", "DMField DMPlex Options", "DM");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-simplex","Create a simplicial DMPlex","ex1.c",simplex,&simplex,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-overlap","DMPlex parallel overlap","ex1.c",overlap,&overlap,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (simplex) {
      PetscBool interpolate = PETSC_TRUE;
      PetscInt  numFaces    = 3;

      ierr = PetscOptionsBegin(comm, "", "DMField DMPlex Simplicial Options", "DM");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-num_faces","Number of edges per direction","ex1.c",numFaces,&numFaces,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-interpolate","Interpolate the DMPlex","ex1.c",interpolate,&interpolate,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
      ierr = DMPlexCreateBoxMesh(comm,dim,numFaces,interpolate,&dm);CHKERRQ(ierr);
      ierr = PetscDTGaussJacobiQuadrature(dim, 1, pointsPerEdge, -1.0, 1.0, &quad);CHKERRQ(ierr);
    } else {
      PetscInt       cells[3] = {3,3,3};
      PetscInt       n = 3, i;
      PetscBool      flags[3];
      PetscInt       inttypes[3];
      DMBoundaryType types[3] = {DM_BOUNDARY_NONE};

      ierr = PetscOptionsBegin(comm, "", "DMField DMPlex Tensor Options", "DM");CHKERRQ(ierr);
      ierr = PetscOptionsIntArray("-cells","Cells per dimension","ex1.c",cells,&n,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEList("-boundary_x","type of boundary in x direction","ex1.c",DMBoundaryTypes,5,DMBoundaryTypes[types[0]],&inttypes[0],&flags[0]);CHKERRQ(ierr);
      ierr = PetscOptionsEList("-boundary_y","type of boundary in y direction","ex1.c",DMBoundaryTypes,5,DMBoundaryTypes[types[1]],&inttypes[1],&flags[1]);CHKERRQ(ierr);
      ierr = PetscOptionsEList("-boundary_z","type of boundary in z direction","ex1.c",DMBoundaryTypes,5,DMBoundaryTypes[types[2]],&inttypes[2],&flags[2]);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);

      for (i = 0; i < 3; i++) {
        if (flags[i]) {types[i] = (DMBoundaryType) inttypes[i];}
      }
      ierr = DMPlexCreateHexBoxMesh(comm,dim,cells,types[0],types[1],types[2],&dm);CHKERRQ(ierr);
      ierr = PetscDTGaussTensorQuadrature(dim, 1, pointsPerEdge, -1.0, 1.0, &quad);CHKERRQ(ierr);
    }
    {
      PetscPartitioner part;
      DM               dmDist;

      ierr = DMPlexGetPartitioner(dm,&part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetType(part,PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
      ierr = DMPlexDistribute(dm,overlap,NULL,&dmDist);CHKERRQ(ierr);
      if (dmDist) {
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm = dmDist;
      }
    }
    ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(dm,dim,nc,simplex,NULL,PETSC_DEFAULT,&fe);CHKERRQ(ierr);
    ierr = DMSetField(dm,0,(PetscObject)fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(dm,&fieldvec);CHKERRQ(ierr);
    {
      PetscErrorCode (*func[1]) (PetscInt,PetscReal,const PetscReal [],PetscInt, PetscScalar *,void *);
      void            *ctxs[1];

      func[0] = radiusSquared;
      ctxs[0] = NULL;

      ierr = DMProjectFunctionLocal(dm,0.0,func,ctxs,INSERT_ALL_VALUES,fieldvec);CHKERRQ(ierr);
    }
    ierr = DMFieldCreateDS(dm,0,fieldvec,&field);CHKERRQ(ierr);
    ierr = VecDestroy(&fieldvec);CHKERRQ(ierr);
  } else if (isda) {
    PetscInt       i;
    PetscScalar    *cv;

    switch (dim) {
    case 1:
      ierr = DMDACreate1d(comm, DM_BOUNDARY_NONE, 3, 1, 1, NULL, &dm);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, &dm);CHKERRQ(ierr);
      break;
    default:
      ierr = DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 3, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, NULL, &dm);CHKERRQ(ierr);
      break;
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr);
    ierr = DMDAGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(nc * (1 << dim),&cv);CHKERRQ(ierr);
    for (i = 0; i < nc * (1 << dim); i++) {
      PetscReal rv;

      ierr = PetscRandomGetValueReal(rand,&rv);CHKERRQ(ierr);
      cv[i] = rv;
    }
    ierr = DMFieldCreateDA(dm,nc,cv,&field);CHKERRQ(ierr);
    ierr = PetscFree(cv);CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim, 1, pointsPerEdge, -1.0, 1.0, &quad);CHKERRQ(ierr);
  } else SETERRQ1(comm,PETSC_ERR_SUP,"This test does not run for DM type %s",type);

  ierr = PetscObjectSetName((PetscObject)dm,"mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)field,"field");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)field,NULL,"-dmfield_view");CHKERRQ(ierr);
  if (numPoint) {ierr = TestEvaluate(field,numPoint,rand);CHKERRQ(ierr);}
  if (numFE) {ierr = TestEvaluateFE(field,numFE,cStart,cEnd,quad,rand);CHKERRQ(ierr);}
  if (numFV) {ierr = TestEvaluateFV(field,numFV,cStart,cEnd,rand);CHKERRQ(ierr);}
  ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
  ierr = DMFieldDestroy(&field);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: da
    requires: !complex
    args: -dm_type da -dim 2 -num_components 2 -num_point_tests 2 -num_fe_tests 2 -num_fv_tests 2 -dmfield_view

  test:
    suffix: da_compact_1
    requires: !complex
    args: -dm_type da -dim 1  -num_fe_tests 2
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: da_compact_2
    requires: !complex
    args: -dm_type da -dim 2  -num_fe_tests 2
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: da_compact_3
    requires: !complex
    args: -dm_type da -dim 3  -num_fe_tests 2
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: ds
    requires: !complex
    args: -dm_type plex -dim 2 -num_components 2  -num_fe_tests 2  -dmfield_view -petscspace_order 2 -num_quad_points 1

  test:
    suffix: ds_compact_simplex_0
    requires: !complex
    args: -dm_type plex -dim 2  -num_fe_tests 2  -petscspace_order 0
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: ds_compact_simplex_1
    requires: !complex
    args: -dm_type plex -dim 2  -num_fe_tests 2  -petscspace_order 1
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: ds_compact_simplex_2
    requires: !complex
    args: -dm_type plex -dim 2  -num_fe_tests 2  -petscspace_order 2
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: ds_compact_tensor_2_0
    requires: !complex
    args: -dm_type plex -dim 2  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_order 0 -simplex 0
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: ds_compact_tensor_2_1
    requires: !complex
    args: -dm_type plex -dim 2  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_order 1 -simplex 0
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: ds_compact_tensor_2_2
    requires: !complex
    args: -dm_type plex -dim 2  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_order 2 -simplex 0
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: ds_compact_tensor_3_0
    requires: !complex
    args: -dm_type plex -dim 3  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_order 0 -simplex 0
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: ds_compact_tensor_3_1
    requires: !complex
    args: -dm_type plex -dim 3  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_order 1 -simplex 0
    test:
      args:
    test:
      args: -test_fe_compact

  test:
    suffix: ds_compact_tensor_3_2
    requires: !complex
    args: -dm_type plex -dim 3  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_order 2 -simplex 0
    test:
      args:
    test:
      args: -test_fe_compact

TEST*/
