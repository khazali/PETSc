static char help[] = "Tests for parallel mesh loading\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

/* List of test meshes

Triangle
--------
Test 0 (2 ranks):
Two triangles sharing a face

        2
      / | \
     /  |  \
    /   |   \
   0  0 | 1  3
    \   |   /
     \  |  /
      \ | /
        1

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3
  cell distribution:
    rank 0: 0
    rank 1: 1

Test 1 (3 ranks):
Four triangles partitioned across 3 ranks

   0 _______ 3
   | \     / |
   |  \ 1 /  |
   |   \ /   |
   | 0  2  2 |
   |   / \   |
   |  / 3 \  |
   | /     \ |
   1 ------- 4

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3
    rank 2: 4
  cell distribution:
    rank 0: 0
    rank 1: 1
    rank 2: 2 3

Tetrahedron
-----------
Test 0:
Two tets sharing a face

 cell   3 _______    cell
 0    / | \      \   1
     /  |  \      \
    /   |   \      \
   0----|----4-----2
    \   |   /      /
     \  |  /      /
      \ | /      /
        1-------
   y
   | x
   |/
   *----z

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3 4
  cell distribution:
    rank 0: 0
    rank 1: 1

Quadrilateral
-------------
Test 0 (2 ranks):
Two quads sharing a face

   3-------2-------5
   |       |       |
   |   0   |   1   |
   |       |       |
   0-------1-------4

  vertex distribution:
    rank 0: 0 1 2
    rank 1: 3 4 5
  cell distribution:
    rank 0: 0
    rank 1: 1

TODO Test 1:
A quad and a triangle sharing a face

   5-------4
   |       | \
   |   0   |  \
   |       | 1 \
   2-------3----6

Hexahedron
----------
Test 0 (2 ranks):
Two hexes sharing a face

cell   7-------------6-------------11 cell
0     /|            /|            /|     1
     / |   F1      / |   F7      / |
    /  |          /  |          /  |
   4-------------5-------------10  |
   |   |     F4  |   |     F10 |   |
   |   |         |   |         |   |
   |F5 |         |F3 |         |F9 |
   |   |  F2     |   |   F8    |   |
   |   3---------|---2---------|---9
   |  /          |  /          |  /
   | /   F0      | /    F6     | /
   |/            |/            |/
   0-------------1-------------8

  vertex distribution:
    rank 0: 0 1 2 3 4 5
    rank 1: 6 7 8 9 10 11
  cell distribution:
    rank 0: 0
    rank 1: 1

*/

typedef struct {
  DM        dm;
  PetscInt  debug;                        /* The debugging level */
  PetscInt  testNum;                      /* Indicates the mesh to create */
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool cellSimplex;                  /* Use simplices or hexes */
  PetscBool distribute;                   /* Distribute the mesh */
  PetscBool iad;                          /* Interpolate the mesh after DMPlexDistribute() */
  PetscBool ibd;                          /* Interpolate the mesh before DMPlexDistribute() */
  PetscBool useGenerator;                 /* Construct mesh with a mesh generator */
  PetscBool hotfix;                       /* Use hotfix for bad edge orientation */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool interpolate = PETSC_FALSE;    /* Interpolate mesh */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dm           = NULL;
  options->debug        = 0;
  options->testNum      = 0;
  options->dim          = 2;
  options->cellSimplex  = PETSC_TRUE;
  options->distribute   = PETSC_FALSE;
  options->iad          = PETSC_FALSE;
  options->useGenerator = PETSC_FALSE;
  options->hotfix       = PETSC_FALSE;
  options->filename[0]  = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex18.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-testnum", "The mesh to create", "ex18.c", options->testNum, &options->testNum, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex18.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex18.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute", "Distribute the mesh", "ex18.c", options->distribute, &options->distribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", "ex18.c", interpolate, &interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate_after_distribute", "Interpolate the mesh only after DMPlexDistribute()", "ex18.c", options->iad, &options->iad, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_generator", "Use a mesh generator to build the mesh", "ex18.c", options->useGenerator, &options->useGenerator, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-hotfix", "", "ex18.c", options->hotfix, &options->hotfix, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex18.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  if (interpolate) {
    options->iad = options->iad && options->distribute;
    options->ibd = !options->iad;
  } else {
    options->ibd = PETSC_FALSE;
    options->iad = PETSC_FALSE;
  }
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSimplex_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {-0.5, 0.5, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {1, 3, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.5, 0.5};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  case 1:
    if (size != 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 3 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {0, 2, 3};
        PetscReal      coords[4] = {0.5, 0.5, 1.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 2:
      {
        const PetscInt numCells  = 2, numVertices = 1, numCorners = 3;
        const int      cells[6]  = {2, 4, 3, 2, 1, 4};
        PetscReal      coords[2] = {1.0, 0.0};
        PetscInt       markerPoints[10] = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSimplex_3D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 4;
        const int      cells[4]  = {0, 2, 1, 3};
        PetscReal      coords[6] = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {1, 2, 4, 3};
        PetscReal      coords[9] = {1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateQuad_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {0, 1, 2, 3};
        PetscReal      coords[6] = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {1, 4, 5, 2};
        PetscReal      coords[6] = {-0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateHex_3D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
    case 0:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const int      cells[8]  = {0, 3, 2, 1, 4, 5, 6, 7};
      PetscReal      coords[6*3] = {-0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0, -0.5,0.0,1.0, 0.0,0.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
    }
    break;
    case 1:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const int      cells[8]  = {1, 2, 9, 8, 5, 10, 11, 6};
      PetscReal      coords[6*3] = {0.0,1.0,1.0, -0.5,1.0,1.0, 0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
    }
    break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CheckMesh(DM dm, AppCtx *user)
{
  PetscReal      detJ, J[9], refVol = 1.0;
  PetscReal      vol;
  PetscInt       dim, depth, d, cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {
    refVol *= 2.0;
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, NULL, J, NULL, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D is inverted, |J| = %g", c, (double)detJ);
    if (user->debug) {PetscPrintf(PETSC_COMM_SELF, "FEM Volume: %g\n", (double)detJ*refVol);CHKERRQ(ierr);}
    if (depth > 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
      if (vol <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %d is inverted, vol = %g", c, (double)vol);
      if (user->debug) {PetscPrintf(PETSC_COMM_SELF, "FVM Volume: %g\n", (double)vol);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AreAllConePointsInArray(DM dm, PetscInt p, PetscInt npoints, const PetscInt *points, PetscBool *flg)
{
  PetscInt i,l,n;
  const PetscInt *cone;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  ierr = DMPlexGetConeSize(dm, p, &n);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscFindInt(cone[i], npoints, points, &l);CHKERRQ(ierr);
    if (l < 0) {
      *flg = PETSC_FALSE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IsAnySupportPointInArray(DM dm, PetscInt p, PetscInt npoints, const PetscInt *points, PetscBool *flg)
{
  PetscInt i,l,n;
  const PetscInt *support;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  ierr = DMPlexGetSupportSize(dm, p, &n);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscFindInt(support[i], npoints, points, &l);CHKERRQ(ierr);
    if (l >= 0) {
      *flg = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CheckPointSF(DM dm, AppCtx *user)
{
  PetscSF sf;
  PetscInt d,depth,i,nleaves,p,plo,phi;
  const PetscInt *locals;
  PetscBool flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, NULL, &nleaves, &locals, NULL);CHKERRQ(ierr);

  /* 1) if some point p is in interface, then all its cone points must be also in interface  */
  for (i=0; i<nleaves; i++) {
    p = locals[i];
    ierr = AreAllConePointsInArray(dm, p, nleaves, locals, &flg);CHKERRQ(ierr);
    if (!flg) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "point SF contains %d but not all points from its cone",p);
  }

  /* 2) if some point p is in interface, then at least one of its support points must lie in interface */
  /* Only check edges & faces */
  for (d=0; d<depth-1; d++) {
    ierr = DMPlexGetDepthStratum(dm, d, &plo, &phi);CHKERRQ(ierr);
    for (i=0; i<nleaves; i++) {
      p = locals[i];
      if (p >= plo && p < phi) {
        ierr = IsAnySupportPointInArray(dm, p, nleaves, locals, &flg);CHKERRQ(ierr);
        if (!flg) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "point SF contains %d but no point from its support",p);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, PetscInt testNum, AppCtx *user, DM *dm)
{
  PetscInt       dim          = user->dim;
  PetscBool      cellSimplex  = user->cellSimplex;
  PetscBool      useGenerator = user->useGenerator;
  PetscBool      interpolate  = user->ibd;
  PetscBool      iad          = user->iad;
  const char    *filename     = user->filename;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  } else if (useGenerator) {
    const PetscInt cells[3] = {2, 2, 2};

    ierr = DMPlexCreateBoxMesh(comm, dim, cellSimplex, cells, NULL, NULL, NULL, interpolate, dm);CHKERRQ(ierr);
  } else {
    switch (dim) {
    case 2:
      if (cellSimplex) {
        ierr = CreateSimplex_2D(comm, interpolate, user, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateQuad_2D(comm, interpolate, user, dm);CHKERRQ(ierr);
      }
      break;
    case 3:
      if (cellSimplex) {
        ierr = CreateSimplex_3D(comm, interpolate, user, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateHex_3D(comm, interpolate, user, dm);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %D", dim);
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Original Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-orig_dm_view");CHKERRQ(ierr);

  if (user->distribute) {
    DM distributedMesh = NULL;
    PetscPartitioner part;

    /* Set partitioner options */
    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);

    /* Redistribute mesh over processes using that partitioner */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
      ierr = PetscObjectSetName((PetscObject) *dm, "Redistributed Mesh");CHKERRQ(ierr);
      ierr = DMViewFromOptions(*dm, NULL, "-dist_dm_view");CHKERRQ(ierr);
    }

    if (iad) {
      DM idm;

      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = idm;
      ierr = PetscObjectSetName((PetscObject) *dm, "Interpolated Redistributed Mesh");CHKERRQ(ierr);
      ierr = DMViewFromOptions(*dm, NULL, "-intp_dm_view");CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Parallel Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, user.testNum, &user, &user.dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSymmetry(user.dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSkeleton(user.dm, user.cellSimplex, 0);CHKERRQ(ierr);
  if (user.ibd || user.iad) {ierr = DMPlexCheckFaces(user.dm, user.cellSimplex, 0);CHKERRQ(ierr);}
  ierr = CheckMesh(user.dm, &user);CHKERRQ(ierr);
  ierr = CheckPointSF(user.dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    nsize: 2
    args: -dm_view ascii::ascii_info_detail

  testset:
    nsize: 2
    args: -interpolate -dm_view ascii::ascii_info_detail
    test:
      suffix: 1
      args: -distribute 0
    test:
      # TODO It would be nice to have the same output here.
      suffix: 1_dist
      args: -distribute 1 -interpolate_after_distribute {{0 1}separate output}
    test:
      suffix: 1_quad
      args: -cell_simplex 0 -distribute 0
    test:
      # TODO It would be nice to have the same output here.
      suffix: 1_quad_dist
      args: -cell_simplex 0 -distribute 1 -interpolate_after_distribute {{0 1}separate output}

  test:
    suffix: 2
    nsize: 3
    args: -testnum 1 -interpolate -dm_view ascii::ascii_info_detail
  test:
    suffix: 3
    nsize: 2
    args: -dim 3 -dm_view ascii::ascii_info_detail

  testset:
    nsize: 2
    args: -dim 3 -interpolate -dm_view ascii::ascii_info_detail -dm_plex_fix_cone_orientation 0
    test:
      # TODO point SF wrong - missing face
      suffix: 4
    test:
      # TODO point SF wrong - shouldn't contain the cell
      suffix: 4_hex
      args: -cell_simplex 0

  testset:
    nsize: 2
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo
    args: -cell_simplex 0 -interpolate -dm_view ascii::ascii_info_detail
    test:
      suffix: 5
      args: -distribute 0
    test:
      suffix: 5_dist
      args: -distribute 1 -interpolate_after_distribute {{0 1}}
TEST*/
