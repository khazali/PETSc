#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateNeighborCSR"
PetscErrorCode DMPlexCreateNeighborCSR(DM dm, PetscInt cellHeight, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency)
{
  const PetscInt maxFaceCases = 30;
  PetscInt       numFaceCases = 0;
  PetscInt       numFaceVertices[30]; /* maxFaceCases, C89 sucks sucks sucks */
  PetscInt      *off, *adj;
  PetscInt      *neighborCells = NULL;
  PetscInt       dim, cellDim, depth = 0, faceDepth, cStart, cEnd, c, numCells, cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* For parallel partitioning, I think you have to communicate supports */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  cellDim = dim - cellHeight;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  if (cEnd - cStart == 0) {
    if (numVertices) *numVertices = 0;
    if (offsets)   *offsets   = NULL;
    if (adjacency) *adjacency = NULL;
    PetscFunctionReturn(0);
  }
  numCells  = cEnd - cStart;
  faceDepth = depth - cellHeight;
  if (dim == depth) {
    PetscInt f, fStart, fEnd;

    ierr = PetscCalloc1(numCells+1, &off);CHKERRQ(ierr);
    /* Count neighboring cells */
    ierr = DMPlexGetHeightStratum(dm, cellHeight+1, &fStart, &fEnd);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *support;
      PetscInt        supportSize;
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      if (supportSize == 2) {
        ++off[support[0]-cStart+1];
        ++off[support[1]-cStart+1];
      }
    }
    /* Prefix sum */
    for (c = 1; c <= numCells; ++c) off[c] += off[c-1];
    if (adjacency) {
      PetscInt *tmp;

      ierr = PetscMalloc1(off[numCells], &adj);CHKERRQ(ierr);
      ierr = PetscMalloc1((numCells+1), &tmp);CHKERRQ(ierr);
      ierr = PetscMemcpy(tmp, off, (numCells+1) * sizeof(PetscInt));CHKERRQ(ierr);
      /* Get neighboring cells */
      for (f = fStart; f < fEnd; ++f) {
        const PetscInt *support;
        PetscInt        supportSize;
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        if (supportSize == 2) {
          adj[tmp[support[0]-cStart]++] = support[1];
          adj[tmp[support[1]-cStart]++] = support[0];
        }
      }
#if defined(PETSC_USE_DEBUG)
      for (c = 0; c < cEnd-cStart; ++c) if (tmp[c] != off[c+1]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offset %d != %d for cell %d", tmp[c], off[c], c+cStart);
#endif
      ierr = PetscFree(tmp);CHKERRQ(ierr);
    }
    if (numVertices) *numVertices = numCells;
    if (offsets)   *offsets   = off;
    if (adjacency) *adjacency = adj;
    PetscFunctionReturn(0);
  }
  /* Setup face recognition */
  if (faceDepth == 1) {
    PetscInt cornersSeen[30] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; /* Could use PetscBT */

    for (c = cStart; c < cEnd; ++c) {
      PetscInt corners;

      ierr = DMPlexGetConeSize(dm, c, &corners);CHKERRQ(ierr);
      if (!cornersSeen[corners]) {
        PetscInt nFV;

        if (numFaceCases >= maxFaceCases) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Exceeded maximum number of face recognition cases");
        cornersSeen[corners] = 1;

        ierr = DMPlexGetNumFaceVertices(dm, cellDim, corners, &nFV);CHKERRQ(ierr);

        numFaceVertices[numFaceCases++] = nFV;
      }
    }
  }
  ierr = PetscCalloc1(numCells+1, &off);CHKERRQ(ierr);
  /* Count neighboring cells */
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt numNeighbors = PETSC_DETERMINE, n;

    ierr = DMPlexGetAdjacency_Internal(dm, cell, PETSC_TRUE, PETSC_FALSE, &numNeighbors, &neighborCells);CHKERRQ(ierr);
    /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
    for (n = 0; n < numNeighbors; ++n) {
      PetscInt        cellPair[2];
      PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
      PetscInt        meetSize = 0;
      const PetscInt *meet    = NULL;

      cellPair[0] = cell; cellPair[1] = neighborCells[n];
      if (cellPair[0] == cellPair[1]) continue;
      if (!found) {
        ierr = DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
        if (meetSize) {
          PetscInt f;

          for (f = 0; f < numFaceCases; ++f) {
            if (numFaceVertices[f] == meetSize) {
              found = PETSC_TRUE;
              break;
            }
          }
        }
        ierr = DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
      }
      if (found) ++off[cell-cStart+1];
    }
  }
  /* Prefix sum */
  for (cell = 1; cell <= numCells; ++cell) off[cell] += off[cell-1];

  if (adjacency) {
    ierr = PetscMalloc1(off[numCells], &adj);CHKERRQ(ierr);
    /* Get neighboring cells */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt numNeighbors = PETSC_DETERMINE, n;
      PetscInt cellOffset   = 0;

      ierr = DMPlexGetAdjacency_Internal(dm, cell, PETSC_TRUE, PETSC_FALSE, &numNeighbors, &neighborCells);CHKERRQ(ierr);
      /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
      for (n = 0; n < numNeighbors; ++n) {
        PetscInt        cellPair[2];
        PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
        PetscInt        meetSize = 0;
        const PetscInt *meet    = NULL;

        cellPair[0] = cell; cellPair[1] = neighborCells[n];
        if (cellPair[0] == cellPair[1]) continue;
        if (!found) {
          ierr = DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
          if (meetSize) {
            PetscInt f;

            for (f = 0; f < numFaceCases; ++f) {
              if (numFaceVertices[f] == meetSize) {
                found = PETSC_TRUE;
                break;
              }
            }
          }
          ierr = DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
        }
        if (found) {
          adj[off[cell-cStart]+cellOffset] = neighborCells[n];
          ++cellOffset;
        }
      }
    }
  }
  ierr = PetscFree(neighborCells);CHKERRQ(ierr);
  if (numVertices) *numVertices = numCells;
  if (offsets)   *offsets   = off;
  if (adjacency) *adjacency = adj;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CHACO)
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
/* Chaco does not have an include file */
PETSC_EXTERN int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);

extern int FREE_GRAPH;

#undef __FUNCT__
#define __FUNCT__ "DMPlexPartition_Chaco"
PetscErrorCode DMPlexPartition_Chaco(DM dm, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection *partSection, IS *partition)
{
  enum {DEFAULT_METHOD = 1, INERTIAL_METHOD = 3};
  MPI_Comm       comm;
  int            nvtxs          = numVertices; /* number of vertices in full graph */
  int           *vwgts          = NULL;   /* weights for all vertices */
  float         *ewgts          = NULL;   /* weights for all edges */
  float         *x              = NULL, *y = NULL, *z = NULL; /* coordinates for inertial method */
  char          *outassignname  = NULL;   /*  name of assignment output file */
  char          *outfilename    = NULL;   /* output file name */
  int            architecture   = 1;      /* 0 => hypercube, d => d-dimensional mesh */
  int            ndims_tot      = 0;      /* total number of cube dimensions to divide */
  int            mesh_dims[3];            /* dimensions of mesh of processors */
  double        *goal          = NULL;    /* desired set sizes for each set */
  int            global_method = 1;       /* global partitioning algorithm */
  int            local_method  = 1;       /* local partitioning algorithm */
  int            rqi_flag      = 0;       /* should I use RQI/Symmlq eigensolver? */
  int            vmax          = 200;     /* how many vertices to coarsen down to? */
  int            ndims         = 1;       /* number of eigenvectors (2^d sets) */
  double         eigtol        = 0.001;   /* tolerance on eigenvectors */
  long           seed          = 123636512; /* for random graph mutations */
  short int     *assignment;              /* Output partition */
  int            fd_stdout, fd_pipe[2];
  PetscInt      *points;
  PetscMPIInt    commSize;
  int            i, v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &commSize);CHKERRQ(ierr);
  if (!numVertices) {
    ierr = PetscSectionCreate(comm, partSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(*partSection, 0, commSize);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, 0, NULL, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
  for (i = 0; i < start[numVertices]; ++i) ++adjacency[i];

  if (global_method == INERTIAL_METHOD) {
    /* manager.createCellCoordinates(nvtxs, &x, &y, &z); */
    SETERRQ(comm, PETSC_ERR_SUP, "Inertial partitioning not yet supported");
  }
  mesh_dims[0] = commSize;
  mesh_dims[1] = 1;
  mesh_dims[2] = 1;
  ierr = PetscMalloc1(nvtxs, &assignment);CHKERRQ(ierr);
  /* Chaco outputs to stdout. We redirect this to a buffer. */
  /* TODO: check error codes for UNIX calls */
#if defined(PETSC_HAVE_UNISTD_H)
  {
    int piperet;
    piperet = pipe(fd_pipe);
    if (piperet) SETERRQ(comm,PETSC_ERR_SYS,"Could not create pipe");
    fd_stdout = dup(1);
    close(1);
    dup2(fd_pipe[1], 1);
  }
#endif
  ierr = interface(nvtxs, (int*) start, (int*) adjacency, vwgts, ewgts, x, y, z, outassignname, outfilename,
                   assignment, architecture, ndims_tot, mesh_dims, goal, global_method, local_method, rqi_flag,
                   vmax, ndims, eigtol, seed);
#if defined(PETSC_HAVE_UNISTD_H)
  {
    char msgLog[10000];
    int  count;

    fflush(stdout);
    count = read(fd_pipe[0], msgLog, (10000-1)*sizeof(char));
    if (count < 0) count = 0;
    msgLog[count] = 0;
    close(1);
    dup2(fd_stdout, 1);
    close(fd_stdout);
    close(fd_pipe[0]);
    close(fd_pipe[1]);
    if (ierr) SETERRQ1(comm, PETSC_ERR_LIB, "Error in Chaco library: %s", msgLog);
  }
#endif
  /* Convert to PetscSection+IS */
  ierr = PetscSectionCreate(comm, partSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*partSection, 0, commSize);CHKERRQ(ierr);
  for (v = 0; v < nvtxs; ++v) {
    ierr = PetscSectionAddDof(*partSection, assignment[v], 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
  ierr = PetscMalloc1(nvtxs, &points);CHKERRQ(ierr);
  for (p = 0, i = 0; p < commSize; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  if (i != nvtxs) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
  ierr = ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  if (global_method == INERTIAL_METHOD) {
    /* manager.destroyCellCoordinates(nvtxs, &x, &y, &z); */
  }
  ierr = PetscFree(assignment);CHKERRQ(ierr);
  for (i = 0; i < start[numVertices]; ++i) --adjacency[i];
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_PARMETIS)
#include <parmetis.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexPartition_ParMetis"
PetscErrorCode DMPlexPartition_ParMetis(DM dm, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection *partSection, IS *partition)
{
  MPI_Comm       comm;
  PetscInt       nvtxs      = numVertices; /* The number of vertices in full graph */
  PetscInt      *vtxdist;                  /* Distribution of vertices across processes */
  PetscInt      *xadj       = start;       /* Start of edge list for each vertex */
  PetscInt      *adjncy     = adjacency;   /* Edge lists for all vertices */
  PetscInt      *vwgt       = NULL;        /* Vertex weights */
  PetscInt      *adjwgt     = NULL;        /* Edge weights */
  PetscInt       wgtflag    = 0;           /* Indicates which weights are present */
  PetscInt       numflag    = 0;           /* Indicates initial offset (0 or 1) */
  PetscInt       ncon       = 1;           /* The number of weights per vertex */
  PetscInt       nparts;                   /* The number of partitions */
  PetscReal     *tpwgts;                   /* The fraction of vertex weights assigned to each partition */
  PetscReal     *ubvec;                    /* The balance intolerance for vertex weights */
  PetscInt       options[5];               /* Options */
  /* Outputs */
  PetscInt       edgeCut;                  /* The number of edges cut by the partition */
  PetscInt      *assignment, *points;
  PetscMPIInt    commSize, rank, p, v, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &commSize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  nparts = commSize;
  options[0] = 0; /* Use all defaults */
  /* Calculate vertex distribution */
  ierr = PetscMalloc4(nparts+1,&vtxdist,nparts*ncon,&tpwgts,ncon,&ubvec,nvtxs,&assignment);CHKERRQ(ierr);
  vtxdist[0] = 0;
  ierr = MPI_Allgather(&nvtxs, 1, MPIU_INT, &vtxdist[1], 1, MPIU_INT, comm);CHKERRQ(ierr);
  for (p = 2; p <= nparts; ++p) {
    vtxdist[p] += vtxdist[p-1];
  }
  /* Calculate weights */
  for (p = 0; p < nparts; ++p) {
    tpwgts[p] = 1.0/nparts;
  }
  ubvec[0] = 1.05;

  if (nparts == 1) {
    ierr = PetscMemzero(assignment, nvtxs * sizeof(PetscInt));
  } else {
    if (vtxdist[1] == vtxdist[nparts]) {
      if (!rank) {
        PetscStackPush("METIS_PartGraphKway");
        ierr = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt, &nparts, tpwgts, ubvec, NULL, &edgeCut, assignment);
        PetscStackPop;
        if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_PartGraphKway()");
      }
    } else {
      PetscStackPush("ParMETIS_V3_PartKway");
      ierr = ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgeCut, assignment, &comm);
      PetscStackPop;
      if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in ParMETIS_V3_PartKway()");
    }
  }
  /* Convert to PetscSection+IS */
  ierr = PetscSectionCreate(comm, partSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*partSection, 0, commSize);CHKERRQ(ierr);
  for (v = 0; v < nvtxs; ++v) {
    ierr = PetscSectionAddDof(*partSection, assignment[v], 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
  ierr = PetscMalloc1(nvtxs, &points);CHKERRQ(ierr);
  for (p = 0, i = 0; p < commSize; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  if (i != nvtxs) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
  ierr = ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  ierr = PetscFree4(vtxdist,tpwgts,ubvec,assignment);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DMPlexEnlargePartition"
/* Expand the partition by BFS on the adjacency graph */
PetscErrorCode DMPlexEnlargePartition(DM dm, const PetscInt start[], const PetscInt adjacency[], PetscSection origPartSection, IS origPartition, PetscSection *partSection, IS *partition)
{
  PetscHashI      h;
  const PetscInt *points;
  PetscInt      **tmpPoints, *newPoints, totPoints = 0;
  PetscInt        pStart, pEnd, part, q;
  PetscBool       useCone;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscHashICreate(h);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), partSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(origPartSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*partSection, pStart, pEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(origPartition, &points);CHKERRQ(ierr);
  ierr = PetscMalloc1((pEnd - pStart), &tmpPoints);CHKERRQ(ierr);
  ierr = DMPlexGetAdjacencyUseCone(dm, &useCone);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseCone(dm, PETSC_TRUE);CHKERRQ(ierr);
  for (part = pStart; part < pEnd; ++part) {
    PetscInt *adj = NULL;
    PetscInt  numPoints, nP, numNewPoints, off, p, n = 0;

    PetscHashIClear(h);
    ierr = PetscSectionGetDof(origPartSection, part, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(origPartSection, part, &off);CHKERRQ(ierr);
    /* Add all existing points to h */
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[off+p];
      PetscHashIAdd(h, point, 1);
    }
    PetscHashISize(h, nP);
    if (nP != numPoints) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid partition has %d points, but only %d were unique", numPoints, nP);
    /* Add all points in next BFS level */
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point   = points[off+p];
      PetscInt       adjSize = PETSC_DETERMINE, a;

      ierr = DMPlexGetAdjacency(dm, point, &adjSize, &adj);CHKERRQ(ierr);
      for (a = 0; a < adjSize; ++a) PetscHashIAdd(h, adj[a], 1);
    }
    PetscHashISize(h, numNewPoints);
    ierr = PetscSectionSetDof(*partSection, part, numNewPoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(numNewPoints, &tmpPoints[part]);CHKERRQ(ierr);
    ierr = PetscHashIGetKeys(h, &n, tmpPoints[part]);CHKERRQ(ierr);
    ierr = PetscFree(adj);CHKERRQ(ierr);
    totPoints += numNewPoints;
  }
  ierr = DMPlexSetAdjacencyUseCone(dm, useCone);CHKERRQ(ierr);
  ierr = ISRestoreIndices(origPartition, &points);CHKERRQ(ierr);
  PetscHashIDestroy(h);
  ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
  ierr = PetscMalloc1(totPoints, &newPoints);CHKERRQ(ierr);
  for (part = pStart, q = 0; part < pEnd; ++part) {
    PetscInt numPoints, p;

    ierr = PetscSectionGetDof(*partSection, part, &numPoints);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p, ++q) newPoints[q] = tmpPoints[part][p];
    ierr = PetscFree(tmpPoints[part]);CHKERRQ(ierr);
  }
  ierr = PetscFree(tmpPoints);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), totPoints, newPoints, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePartition"
/*
  DMPlexCreatePartition - Create a non-overlapping partition of the points at the given height

  Collective on DM

  Input Parameters:
  + dm - The DM
  . height - The height for points in the partition
  - enlarge - Expand each partition with neighbors

  Output Parameters:
  + partSection - The PetscSection giving the division of points by partition
  . partition - The list of points by partition
  . origPartSection - If enlarge is true, the PetscSection giving the division of points before enlarging by partition, otherwise NULL
  - origPartition - If enlarge is true, the list of points before enlarging by partition, otherwise NULL

  Level: developer

.seealso DMPlexDistribute()
*/
PetscErrorCode DMPlexCreatePartition(DM dm, const char name[], PetscInt height, PetscBool enlarge, PetscSection *partSection, IS *partition, PetscSection *origPartSection, IS *origPartition)
{
  char           partname[1024];
  PetscBool      isChaco = PETSC_FALSE, isMetis = PETSC_FALSE, flg;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);

  if (origPartSection) *origPartSection = NULL;
  if (origPartition) *origPartition   = NULL;
  if (size == 1) {
    PetscInt *points;
    PetscInt  cStart, cEnd, c;

    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), partSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(*partSection, 0, size);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*partSection, 0, cEnd-cStart);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
    ierr = PetscMalloc1((cEnd - cStart), &points);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) points[c] = c;
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), cEnd-cStart, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscOptionsGetString(((PetscObject) dm)->prefix, "-dm_plex_partitioner", partname, 1024, &flg);CHKERRQ(ierr);
  if (flg) name = partname;
  if (name) {
    ierr = PetscStrcmp(name, "chaco", &isChaco);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "metis", &isMetis);CHKERRQ(ierr);
  }
  if (height == 0) {
    PetscInt  numVertices;
    PetscInt *start     = NULL;
    PetscInt *adjacency = NULL;

    ierr = DMPlexCreateNeighborCSR(dm, 0, &numVertices, &start, &adjacency);CHKERRQ(ierr);
    if (!name || isChaco) {
#if defined(PETSC_HAVE_CHACO)
      ierr = DMPlexPartition_Chaco(dm, numVertices, start, adjacency, partSection, partition);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-chaco.");
#endif
    } else if (isMetis) {
#if defined(PETSC_HAVE_PARMETIS)
      ierr = DMPlexPartition_ParMetis(dm, numVertices, start, adjacency, partSection, partition);CHKERRQ(ierr);
#endif
    } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Unknown mesh partitioning package %s", name);
    if (enlarge) {
      *origPartSection = *partSection;
      *origPartition   = *partition;

      ierr = DMPlexEnlargePartition(dm, start, adjacency, *origPartSection, *origPartition, partSection, partition);CHKERRQ(ierr);
    }
    ierr = PetscFree(start);CHKERRQ(ierr);
    ierr = PetscFree(adjacency);CHKERRQ(ierr);
# if 0
  } else if (height == 1) {
    /* Build the dual graph for faces and partition the hypergraph */
    PetscInt numEdges;

    buildFaceCSRV(mesh, mesh->getFactory()->getNumbering(mesh, mesh->depth()-1), &numEdges, &start, &adjacency, GraphPartitioner::zeroBase());
    GraphPartitioner().partition(numEdges, start, adjacency, partition, manager);
    destroyCSR(numEdges, start, adjacency);
#endif
  } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid partition height %D", height);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePartitionClosure"
PetscErrorCode DMPlexCreatePartitionClosure(DM dm, PetscSection pointSection, IS pointPartition, PetscSection *section, IS *partition)
{
  /* const PetscInt  height = 0; */
  const PetscInt *partArray;
  PetscInt       *allPoints, *packPoints;
  PetscInt        rStart, rEnd, rank, pStart, pEnd, newSize;
  PetscErrorCode  ierr;
  PetscBT         bt;
  PetscSegBuffer  segpack,segpart;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(pointSection, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(pointPartition, &partArray);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, rStart, rEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscBTCreate(pEnd-pStart,&bt);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscInt),1000,&segpack);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscInt),1000,&segpart);CHKERRQ(ierr);
  for (rank = rStart; rank < rEnd; ++rank) {
    PetscInt partSize = 0, numPoints, offset, p, *PETSC_RESTRICT placePoints;

    ierr = PetscSectionGetDof(pointSection, rank, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pointSection, rank, &offset);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      PetscInt  point   = partArray[offset+p], closureSize, c;
      PetscInt *closure = NULL;

      /* TODO Include support for height > 0 case */
      ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (c=0; c<closureSize; c++) {
        PetscInt cpoint = closure[c*2];
        if (!PetscBTLookupSet(bt,cpoint-pStart)) {
          PetscInt *PETSC_RESTRICT pt;
          partSize++;
          ierr = PetscSegBufferGetInts(segpart,1,&pt);CHKERRQ(ierr);
          *pt = cpoint;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetDof(*section, rank, partSize);CHKERRQ(ierr);
    ierr = PetscSegBufferGetInts(segpack,partSize,&placePoints);CHKERRQ(ierr);
    ierr = PetscSegBufferExtractTo(segpart,placePoints);CHKERRQ(ierr);
    ierr = PetscSortInt(partSize,placePoints);CHKERRQ(ierr);
    for (p=0; p<partSize; p++) {ierr = PetscBTClear(bt,placePoints[p]-pStart);CHKERRQ(ierr);}
  }
  ierr = PetscBTDestroy(&bt);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&segpart);CHKERRQ(ierr);

  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(*section, &newSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(newSize, &allPoints);CHKERRQ(ierr);

  ierr = PetscSegBufferExtractInPlace(segpack,&packPoints);CHKERRQ(ierr);
  for (rank = rStart; rank < rEnd; ++rank) {
    PetscInt numPoints, offset;

    ierr = PetscSectionGetDof(*section, rank, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(*section, rank, &offset);CHKERRQ(ierr);
    ierr = PetscMemcpy(&allPoints[offset], packPoints, numPoints * sizeof(PetscInt));CHKERRQ(ierr);
    packPoints += numPoints;
  }

  ierr = PetscSegBufferDestroy(&segpack);CHKERRQ(ierr);
  ierr = ISRestoreIndices(pointPartition, &partArray);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), newSize, allPoints, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePartitionOverlap_AddRemoteConnections"
PetscErrorCode DMPlexCreatePartitionOverlap_AddRemoteConnections(DM dm, PetscInt *recvRanks, DMLabel label)
{
  MPI_Comm           comm;
  PetscMPIInt        rank, numProcs;
  PetscSF            pointSF, rankSF, remoteSF;
  PetscSection       localSection, remoteSection;
  ISLocalToGlobalMapping renumbering;
  PetscInt           j, p, pStart, pEnd, proc, dof, offset, localSize, remoteSize;
  PetscInt          *localIdx, *remoteOffsets;
  PetscBool          assigned;
  const PetscInt    *ltog;
  PetscSFNode       *localConnections, *remoteConnections, *remoteRanks;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);

  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);

  /* We need to inform point owners of overlap connections made on the local
     process. For this we list all locally made connections for each root owner
     in the pointSF, where each connection is an SFNode detailing which point to
     donate to which process. */
  ierr = PetscSectionCreate(comm, &localSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(localSection, 0, numProcs);CHKERRQ(ierr);
  for (p=pStart; p<pEnd; p++) {
    if (recvRanks[p] < 0) continue;
    for (proc=0; proc<numProcs; proc++) {
      ierr = DMLabelStratumHasPoint(label, proc, p, &assigned);CHKERRQ(ierr);
      if (assigned) {
        ierr = PetscSectionGetDof(localSection, recvRanks[p], &dof);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(localSection, recvRanks[p], dof+1);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(localSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(localSection, &localSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(localSize, &localConnections);CHKERRQ(ierr);
  ierr = PetscCalloc1(numProcs, &localIdx);CHKERRQ(ierr);
  for (p=pStart; p<pEnd; p++) {
    if (recvRanks[p] < 0) continue;
    for (proc=0; proc<numProcs; proc++) {
      ierr = DMLabelStratumHasPoint(label, proc, p, &assigned);CHKERRQ(ierr);
      if (assigned) {
        ierr = PetscSectionGetOffset(localSection, recvRanks[p], &offset);CHKERRQ(ierr);
        localConnections[offset + localIdx[recvRanks[p]]].index = p;
        localConnections[offset + localIdx[recvRanks[p]]].rank = proc;
        localIdx[recvRanks[p]]++;
      }
    }
  }

  /* Translate local connection indices to remote numbering */
  ierr = ISLocalToGlobalMappingCreateSF(pointSF, 0, &renumbering);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(renumbering, &ltog);CHKERRQ(ierr);
  for (proc=0; proc<numProcs; proc++) {
    ierr = PetscSectionGetDof(localSection, proc, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(localSection, proc, &offset);CHKERRQ(ierr);
    for (j=0; j<dof; j++) {
      localConnections[offset+j].index = ltog[ localConnections[offset+j].index ];
    }
  }

  /* Build SF that maps into local section entries */
  ierr = PetscMalloc1(numProcs, &remoteRanks);CHKERRQ(ierr);
  for (p = 0; p < numProcs; p++) {
    remoteRanks[p].rank = p;
    remoteRanks[p].index = rank;
  }
  ierr = PetscSFCreate(comm, &rankSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(rankSF, numProcs, numProcs, NULL, PETSC_OWN_POINTER, remoteRanks, PETSC_OWN_POINTER);CHKERRQ(ierr);

  /* Push local overlap connections to the point owner */
  ierr = PetscSectionCreate(comm, &remoteSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(rankSF, localSection, &remoteOffsets, remoteSection);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(rankSF, localSection, remoteOffsets, remoteSection, &remoteSF);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(remoteSection, &remoteSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(remoteSize, &remoteConnections);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(remoteSF, MPIU_2INT, localConnections, remoteConnections);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(remoteSF, MPIU_2INT, localConnections, remoteConnections);CHKERRQ(ierr);

  /* Add remote connections to label */
  for (p=0; p<remoteSize; p++) {
    proc = remoteConnections[p].rank;
    if (proc != rank) {
      ierr = DMLabelSetValue(label, remoteConnections[p].index, proc);
    }
  }

  ierr = PetscSFDestroy(&rankSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&remoteSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&localSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&remoteSection);CHKERRQ(ierr);
  ierr = PetscFree3(localConnections, remoteConnections, localIdx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexCreatePartitionOverlap - Create a partition of local cells in the remote overlap of neightbouring processes

  Collective on DM

  Input Parameters:
  + dm - The DM

  Output Parameters:
  + section - The PetscSection giving the division of overlap cells by partition
  . overlap - The list of overlap cells by partition

  The user can control the definition of adjacency for the mesh using DMPlexGetAdjacencyUseCone() and
  DMPlexSetAdjacencyUseClosure(). They should choose the combination appropriate for the function
  representation on the mesh.

  Level: intermediate

.seealso DMPlexDistribute(), DMPlexCreatePartition(), DMPlexCreatePartitionClosure(), DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure()
*/
#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePartitionOverlap"
PetscErrorCode DMPlexCreatePartitionOverlap(DM dm, PetscSection *section, IS *overlap)
{
  PetscMPIInt        rank, numProcs;
  MPI_Comm           comm;
  PetscSF            pointSF;
  DMLabel            label;
  const PetscInt    *ilocal, *sendRanks;
  const PetscSFNode *iremote;
  PetscSF            sharedRankSF;
  IS                 ranks;
  PetscSection       sendRankSection, sharedRankSection;
  PetscInt           p, pStart, pEnd, a, adjSize, nleaves;
  PetscInt           numRanks, proc, sharedRankSize, offset;
  PetscInt          *recvRanks, *remoteOffsets, *sharedRanks, *adjacency=NULL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);

  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMLabelCreate("overlap", &label);CHKERRQ(ierr);

  /* The local contribution to the overlap region of neighbouring processes
     is computed by adding the locally known points adjacenct to a point
     that is connected to a neighbour to a label, where the label value
     is the neighbouring process ID. Threed types of connections are
     considered:
       * Direct connections via the leaf->root mapping in the pointSF
       * Shared-point connections to other receivers of the same root
       * Remote connections to partitions that we don't share any
         points with, but are connected to via an intermediary process
   */

  /* Build a root->leaf-rank mapping and push this over the pointSF
     to establish connections via shared roots */
  ierr = PetscSFCreateSendRanks(pointSF, &sendRankSection, &ranks);CHKERRQ(ierr);
  ierr = ISGetIndices(ranks, &sendRanks);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &sharedRankSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, sendRankSection, &remoteOffsets, sharedRankSection);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(pointSF, sendRankSection, remoteOffsets, sharedRankSection, &sharedRankSF);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(sharedRankSection, &sharedRankSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(sharedRankSize, &sharedRanks);
  ierr = PetscSFBcastBegin(sharedRankSF, MPIU_INT, sendRanks, sharedRanks);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sharedRankSF, MPIU_INT, sendRanks, sharedRanks);CHKERRQ(ierr);

  /* Add direct and shared-point connections via our local leaves */
  ierr = PetscSFGetGraph(pointSF, NULL, &nleaves, &ilocal, &iremote);CHKERRQ(ierr);
  for (p=0; p<nleaves; p++) {
    adjSize = PETSC_DETERMINE;
    ierr = DMPlexGetAdjacency(dm, ilocal[p], &adjSize, &adjacency);CHKERRQ(ierr);
    /* Add connections via our leaves to the root owner */
    for (a=0; a<adjSize; a++) {
      ierr = DMLabelSetValue(label, adjacency[a], iremote[p].rank);CHKERRQ(ierr);
    }
    /* Add connections to other receivers of the shared root */
    ierr = PetscSectionGetDof(sharedRankSection, ilocal[p], &numRanks);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sharedRankSection, ilocal[p], &offset);CHKERRQ(ierr);
    for (proc=0; proc<numRanks; proc++) {
      if (sharedRanks[offset+proc] == rank) continue;
      for (a=0; a<adjSize; a++) {
        ierr = DMLabelSetValue(label, adjacency[a], sharedRanks[offset+proc]);CHKERRQ(ierr);
      }
    }
  }
  /* Add connections via roots to leaf owners */
  for (p=pStart; p<pEnd; p++) {
    ierr = PetscSectionGetDof(sendRankSection, p, &numRanks);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sendRankSection, p, &offset);CHKERRQ(ierr);
    if (numRanks <= 0) continue;
    adjSize = PETSC_DETERMINE;
    ierr = DMPlexGetAdjacency(dm, p, &adjSize, &adjacency);CHKERRQ(ierr);
    for (proc=0; proc<numRanks; proc++) {
      for (a=0; a<adjSize; a++) {
        ierr = DMLabelSetValue(label, adjacency[a], sendRanks[offset+proc]);CHKERRQ(ierr);
      }
    }
  }

  /* Before removing non-owned points we need to inform root owners
     and receiving process of all remote connections we established. */
  ierr = PetscSFGetGraph(pointSF, NULL, &nleaves, &ilocal, &iremote);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &recvRanks);CHKERRQ(ierr);
  for (p=pStart; p<pEnd; p++) recvRanks[p] = -1;
  for (p=0; p<nleaves; p++) recvRanks[ilocal[p]] = iremote[p].rank;
  ierr = DMPlexCreatePartitionOverlap_AddRemoteConnections(dm, recvRanks, label);CHKERRQ(ierr);

  /* Strip points we don't own */
  for (p=0; p<nleaves; p++) {
    for (proc=0; proc<numProcs; proc++) {
      ierr = DMLabelClearValue(label, ilocal[p], proc);CHKERRQ(ierr);
    }
  }
  /* Strip points we already send via the pointSF */
  for (p=pStart; p<pEnd; p++) {
    ierr = PetscSectionGetDof(sendRankSection, p, &numRanks);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sendRankSection, p, &offset);CHKERRQ(ierr);
    for (proc=0; proc<numRanks; proc++) {
      ierr = DMLabelClearValue(label, p, sendRanks[offset+proc]);CHKERRQ(ierr);
    }
  }
  /* Build a partition (section/IS pair) from the label */
  ierr = DMPlexCreateLabelPartition(dm, label, section, overlap);CHKERRQ(ierr);

  /* Clean up */
  ierr = ISRestoreIndices(ranks, &sendRanks);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sendRankSection);CHKERRQ(ierr);
  ierr = ISDestroy(&ranks);CHKERRQ(ierr);
  ierr = PetscFree(sharedRanks);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePartitionSF"
PetscErrorCode DMPlexCreatePartitionSF(DM dm, PetscSection section, IS partition, PetscSF *sf)
{
  MPI_Comm           comm;
  PetscMPIInt        rank, numProcs;
  PetscInt           p, pStart, pEnd, nleaves, *remoteOffsets, *remotePoints;
  PetscSF            rankSF, partitionSF;
  PetscSFNode       *iremoteRanks, *iremote;
  const PetscInt    *partitionArray;
  const PetscSFNode *premote;
  PetscSection       leafSection;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);

  /* Build SF that maps into section entries */
  ierr = PetscMalloc1(numProcs, &iremoteRanks);CHKERRQ(ierr);
  for (p = 0; p < numProcs; p++) {
    iremoteRanks[p].rank = p;
    iremoteRanks[p].index = rank;
  }
  ierr = PetscSFCreate(comm, &rankSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(rankSF, numProcs, numProcs, NULL, PETSC_OWN_POINTER, iremoteRanks, PETSC_OWN_POINTER);CHKERRQ(ierr);

  /* Build SF that maps over the partition IS */
  ierr = PetscSectionCreate(comm, &leafSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(rankSF, section, &remoteOffsets, leafSection);CHKERRQ(ierr);
  ierr = PetscSFCreateRemoteOffsets(rankSF, section, leafSection, &remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(rankSF, section, remoteOffsets, leafSection, &partitionSF);CHKERRQ(ierr);

  /* Derive points from the remote partition IS */
  ierr = PetscSFGetGraph(partitionSF, NULL, &nleaves, NULL, &premote);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves, &remotePoints);CHKERRQ(ierr);
  ierr = ISGetIndices(partition, &partitionArray);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(partitionSF, MPIU_INT, partitionArray, remotePoints);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(partitionSF, MPIU_INT, partitionArray, remotePoints);CHKERRQ(ierr);

  /* Build the new SF mapping actual remote roots into local leafs */
  ierr = PetscSFCreate(comm, sf);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves, &iremote);CHKERRQ(ierr);
  for (p=0; p<nleaves; p++) {
    iremote[p].index = remotePoints[p];
    iremote[p].rank = premote[p].rank;
  }
  ierr = PetscSFSetGraph(*sf, pEnd-pStart, nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);

  ierr = PetscFree(remotePoints);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&rankSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&partitionSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafSection);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateOverlapMigrationSF"
PetscErrorCode DMPlexCreateOverlapMigrationSF(DM dm, PetscSF overlapSF, PetscSF *migrationSF)
{
  MPI_Comm           comm;
  PetscMPIInt        rank, numProcs;
  PetscInt           d, dim, p, pStart, pEnd, nroots, nleaves, newLeaves, point, numSharedPoints;
  PetscInt          *pointDepths, *remoteDepths, *ilocal;
  PetscSFNode       *iremote;
  PetscSF            pointSF;
  const PetscInt    *sharedLocal;
  const PetscSFNode *overlapRemote, *sharedRemote;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  /* Before building the migration SF we need to know the new stratum offsets */
  ierr = PetscSFGetGraph(overlapSF, &nroots, &nleaves, NULL, &overlapRemote);CHKERRQ(ierr);
  ierr = PetscMalloc1(nroots, &pointDepths);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves, &remoteDepths);CHKERRQ(ierr);
  for (d=0; d<dim+1; d++) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p=pStart; p<pEnd; p++) pointDepths[p] = d;
  }
  for (p=0; p<nleaves; p++) remoteDepths[p] = -1;
  ierr = PetscSFBcastBegin(overlapSF, MPIU_INT, pointDepths, remoteDepths);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(overlapSF, MPIU_INT, pointDepths, remoteDepths);CHKERRQ(ierr);

  /* Count recevied points in each stratum and compute the internal strata shift */
  PetscInt depthRecv[dim+1], depthShift[dim+1], depthIdx[dim+1];
  for (d=0; d<dim+1; d++) depthRecv[d]=0;
  for (p=0; p<nleaves; p++) depthRecv[remoteDepths[p]]++;
  depthShift[dim] = 0;
  for (d=0; d<dim; d++) depthShift[d] = depthRecv[dim];
  for (d=1; d<dim; d++) depthShift[d] += depthRecv[0];
  for (d=dim-2; d>0; d--) depthShift[d] += depthRecv[d+1];
  for (d=0; d<dim+1; d++) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    depthIdx[d] = pStart + depthShift[d];
  }

  /* Form the overlap SF build an SF that describes the full overlap migration SF */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  newLeaves = pEnd - pStart + nleaves;
  ierr = PetscMalloc2(newLeaves, &ilocal, newLeaves, &iremote);CHKERRQ(ierr);
  /* First map local points to themselves */
  for (d=0; d<dim+1; d++) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p=pStart; p<pEnd; p++) {
      point = p + depthShift[d];
      ilocal[point] = point;
      iremote[point].index = p;
      iremote[point].rank = rank;
      depthIdx[d]++;
    }
  }

  /* Add in the remote roots for currently shared points */
  ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(pointSF, NULL, &numSharedPoints, &sharedLocal, &sharedRemote);CHKERRQ(ierr);
  for (d=0; d<dim+1; d++) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p=0; p<numSharedPoints; p++) {
      if (pStart <= sharedLocal[p] && sharedLocal[p] < pEnd) {
        point = sharedLocal[p] + depthShift[d];
        iremote[point].index = sharedRemote[p].index;
        iremote[point].rank = sharedRemote[p].rank;
      }
    }
  }

  /* Now add the incoming overlap points */
  for (p=0; p<nleaves; p++) {
    point = depthIdx[remoteDepths[p]];
    ilocal[point] = point;
    iremote[point].index = overlapRemote[p].index;
    iremote[point].rank = overlapRemote[p].rank;
    depthIdx[remoteDepths[p]]++;
  }

  ierr = PetscSFCreate(comm, migrationSF);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*migrationSF, pEnd-pStart, newLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
