static char help[] = "Demonstration of creating and viewing DMFields objects.\n\n";

#include <petscdmfield.h>
#include <petscdmplex.h>
#include <petscdmda.h>

static PetscErrorCode TestEvaluate(DMField field, PetscRandom rand)
{
  DM             dm;
  PetscInt       dim, i, nc;
  PetscInt       n = 2;
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
  ierr = PetscOptionsBegin(comm, "", "DMField TestEvaluate Options", "DM");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_points", "Number of test points", "ex1.c", n, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)field),n * dim,PETSC_DETERMINE,&points);CHKERRQ(ierr);
  ierr = VecSetBlockSize(points,dim);CHKERRQ(ierr);
  ierr = VecGetArray(points,&array);CHKERRQ(ierr);
  for (i = 0; i < n * dim; i++) {ierr = PetscRandomGetValue(rand,&array[i]);CHKERRQ(ierr);}
  ierr = VecRestoreArray(points,&array);CHKERRQ(ierr);
  ierr = PetscMalloc6(n*nc,&B,n*nc,&rB,n*nc*dim,&D,n*nc*dim,&rD,n*nc*dim*dim,&H,n*nc*dim*dim,&rH);CHKERRQ(ierr);
  ierr = DMFieldEvaluate(field,points,B,D,H);CHKERRQ(ierr);
  ierr = DMFieldEvaluateReal(field,points,rB,rD,rH);CHKERRQ(ierr);
  viewer = PETSC_VIEWER_STDOUT_(comm);

  ierr = PetscObjectSetName((PetscObject)points,"Test Points");CHKERRQ(ierr);
  ierr = VecView(points,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"B:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(n*nc,B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"D:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(n*nc*dim,D,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"H:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(n*nc*dim*dim,H,viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"rB:\n");CHKERRQ(ierr);
  ierr = PetscRealView(n*nc,rB,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rD:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(n*nc*dim,rD,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rH:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(n*nc*dim*dim,rH,viewer);CHKERRQ(ierr);

  ierr = PetscFree6(B,rB,D,rD,H,rH);CHKERRQ(ierr);
  ierr = VecDestroy(&points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEvaluateFE(DMField field, PetscInt cStart, PetscInt cEnd, PetscQuadrature quad, PetscRandom rand)
{
  DM             dm;
  PetscInt       dim, i, nc, nq;
  PetscInt       n = 2, N;
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
  ierr = PetscOptionsBegin(comm, "", "DMField TestEvaluateFE Options", "DM");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_fe_cells", "Number of test cells for FE", "ex1.c", n, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
  ierr = DMFieldEvaluateFE(field,n,cells,quad,B,D,H);CHKERRQ(ierr);
  ierr = DMFieldEvaluateFEReal(field,n,cells,quad,rB,rD,rH);CHKERRQ(ierr);
  viewer = PETSC_VIEWER_STDOUT_(comm);

  ierr = PetscObjectSetName((PetscObject)quad,"Test quadrature");CHKERRQ(ierr);
  ierr = PetscQuadratureView(quad,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Test Cells:\n");CHKERRQ(ierr);
  ierr = PetscIntView(n,cells,viewer);CHKERRQ(ierr);
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

  ierr = PetscFree6(B,rB,D,rD,H,rH);CHKERRQ(ierr);
  ierr = PetscFree(cells);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMField Tutorial Options", "DM");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_type","DM implementation on which to define field","ex1.c",DMList,type,type,256,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","DM intrinsic dimension", "ex1.c", dim, &dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_components","Number of components in field", "ex1.c", nc, &nc, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_quad_points","Number of quadrature points per dimension", "ex1.c", pointsPerEdge, &pointsPerEdge, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (dim > 3) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"This examples works for dim <= 3, not %D",dim);
  ierr = PetscStrncmp(type,DMPLEX,256,&isplex);CHKERRQ(ierr);
  ierr = PetscStrncmp(type,DMDA,256,&isda);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  if (isplex) {
    PetscBool simplex = PETSC_TRUE;
    PetscInt  overlap = 0;

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
  ierr = TestEvaluate(field,rand);CHKERRQ(ierr);
  ierr = TestEvaluateFE(field,cStart,cEnd,quad,rand);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
  ierr = DMFieldDestroy(&field);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
