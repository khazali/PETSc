#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetAdjacencyUseCone"
/*@
  DMPlexSetAdjacencyUseCone - Define adjacency in the mesh using either the cone or the support first

  Input Parameters:
+ dm      - The DM object
- useCone - Flag to use the cone first

  Level: intermediate

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)), useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)), useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

.seealso: DMPlexGetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure(), DMPlexGetAdjacencyUseClosure(), DMPlexDistribute(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexSetAdjacencyUseCone(DM dm, PetscBool useCone)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->useCone = useCone;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacencyUseCone"
/*@
  DMPlexGetAdjacencyUseCone - Query whether adjacency in the mesh uses the cone or the support first

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. useCone - Flag to use the cone first

  Level: intermediate

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)), useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)), useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

.seealso: DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure(), DMPlexGetAdjacencyUseClosure(), DMPlexDistribute(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexGetAdjacencyUseCone(DM dm, PetscBool *useCone)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(useCone, 2);
  *useCone = mesh->useCone;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetAdjacencyUseClosure"
/*@
  DMPlexSetAdjacencyUseClosure - Define adjacency in the mesh using the transitive closure

  Input Parameters:
+ dm      - The DM object
- useClosure - Flag to use the closure

  Level: intermediate

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)), useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)), useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

.seealso: DMPlexGetAdjacencyUseClosure(), DMPlexSetAdjacencyUseCone(), DMPlexGetAdjacencyUseCone(), DMPlexDistribute(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexSetAdjacencyUseClosure(DM dm, PetscBool useClosure)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->useClosure = useClosure;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacencyUseClosure"
/*@
  DMPlexGetAdjacencyUseClosure - Query whether adjacency in the mesh uses the transitive closure

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. useClosure - Flag to use the closure

  Level: intermediate

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)), useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)), useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

.seealso: DMPlexSetAdjacencyUseClosure(), DMPlexSetAdjacencyUseCone(), DMPlexGetAdjacencyUseCone(), DMPlexDistribute(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexGetAdjacencyUseClosure(DM dm, PetscBool *useClosure)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(useClosure, 2);
  *useClosure = mesh->useClosure;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetAdjacencyUseConstraints"
/*@
  DMPlexSetAdjacencyUseConstraints - Define adjacency in the mesh using the point-to-point constraints.

  Input Parameters:
+ dm      - The DM object
- useConstraints - Flag to use the constraints.  If PETSC_TRUE, then constrained points are omitted from DMPlexGetAdjacency(), and their anchor points appear in their place.

  Level: intermediate

.seealso: DMPlexGetAdjacencyUseClosure(), DMPlexSetAdjacencyUseCone(), DMPlexGetAdjacencyUseCone(), DMPlexDistribute(), DMPlexPreallocateOperator(), DMPlexSetConstraints()
@*/
PetscErrorCode DMPlexSetAdjacencyUseConstraints(DM dm, PetscBool useConstraints)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->useConstraints = useConstraints;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacencyUseConstraints"
/*@
  DMPlexGetAdjacencyUseConstraints - Query whether adjacency in the mesh uses the point-to-point constraints.

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. useConstraints - Flag to use the closure.  If PETSC_TRUE, then constrained points are omitted from DMPlexGetAdjacency(), and their anchor points appear in their place.

  Level: intermediate

.seealso: DMPlexSetAdjacencyUseConstraints(), DMPlexSetAdjacencyUseCone(), DMPlexGetAdjacencyUseCone(), DMPlexDistribute(), DMPlexPreallocateOperator(), DMPlexSetConstraints()
@*/
PetscErrorCode DMPlexGetAdjacencyUseConstraints(DM dm, PetscBool *useConstraints)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(useConstraints, 2);
  *useConstraints = mesh->useConstraints;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency_Cone_Internal"
static PetscErrorCode DMPlexGetAdjacency_Cone_Internal(DM dm, PetscInt p, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *cone = NULL;
  PetscInt        numAdj = 0, maxAdjSize = *adjSize, coneSize, c;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  for (c = 0; c < coneSize; ++c) {
    const PetscInt *support = NULL;
    PetscInt        supportSize, s, q;

    ierr = DMPlexGetSupportSize(dm, cone[c], &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, cone[c], &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      for (q = 0; q < numAdj || (adj[numAdj++] = support[s],0); ++q) {
        if (support[s] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
  }
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency_Support_Internal"
static PetscErrorCode DMPlexGetAdjacency_Support_Internal(DM dm, PetscInt p, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *support = NULL;
  PetscInt        numAdj   = 0, maxAdjSize = *adjSize, supportSize, s;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
  for (s = 0; s < supportSize; ++s) {
    const PetscInt *cone = NULL;
    PetscInt        coneSize, c, q;

    ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
    for (c = 0; c < coneSize; ++c) {
      for (q = 0; q < numAdj || (adj[numAdj++] = cone[c],0); ++q) {
        if (cone[c] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
  }
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency_Transitive_Internal"
static PetscErrorCode DMPlexGetAdjacency_Transitive_Internal(DM dm, PetscInt p, PetscBool useClosure, PetscInt *adjSize, PetscInt adj[])
{
  PetscInt      *star = NULL;
  PetscInt       numAdj = 0, maxAdjSize = *adjSize, starSize, s;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetTransitiveClosure(dm, p, useClosure, &starSize, &star);CHKERRQ(ierr);
  for (s = 0; s < starSize*2; s += 2) {
    const PetscInt *closure = NULL;
    PetscInt        closureSize, c, q;

    ierr = DMPlexGetTransitiveClosure(dm, star[s], (PetscBool)!useClosure, &closureSize, (PetscInt**) &closure);CHKERRQ(ierr);
    for (c = 0; c < closureSize*2; c += 2) {
      for (q = 0; q < numAdj || (adj[numAdj++] = closure[c],0); ++q) {
        if (closure[c] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, star[s], (PetscBool)!useClosure, &closureSize, (PetscInt**) &closure);CHKERRQ(ierr);
  }
  ierr = DMPlexRestoreTransitiveClosure(dm, p, useClosure, &starSize, &star);CHKERRQ(ierr);
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency_Internal"
PetscErrorCode DMPlexGetAdjacency_Internal(DM dm, PetscInt p, PetscBool useCone, PetscBool useTransitiveClosure, PetscBool useConstraints, PetscInt *adjSize, PetscInt *adj[])
{
  static PetscInt asiz = 0;
  PetscInt maxAnchors = 1;
  PetscInt aStart = -1, aEnd = -1;
  PetscInt maxAdjSize;
  PetscSection aSec = NULL;
  IS aIS = NULL;
  const PetscInt *anchors;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  if (useConstraints) {
    ierr = DMPlexGetConstraints(dm,&aSec,&aIS);CHKERRQ(ierr);
    if (aSec) {
      ierr = PetscSectionGetMaxDof(aSec,&maxAnchors);CHKERRQ(ierr);
      maxAnchors = PetscMax(1,maxAnchors);
      ierr = PetscSectionGetChart(aSec,&aStart,&aEnd);CHKERRQ(ierr);
      ierr = ISGetIndices(aIS,&anchors);CHKERRQ(ierr);
    }
  }
  if (!*adj) {
    PetscInt depth, coneSeries, supportSeries, maxC, maxS, pStart, pEnd;

    ierr  = DMPlexGetChart(dm, &pStart,&pEnd);CHKERRQ(ierr);
    ierr  = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr  = DMPlexGetMaxSizes(dm, &maxC, &maxS);CHKERRQ(ierr);
    coneSeries    = (maxC > 1) ? ((PetscPowInt(maxC,depth+1)-1)/(maxC-1)) : depth+1;
    supportSeries = (maxS > 1) ? ((PetscPowInt(maxS,depth+1)-1)/(maxS-1)) : depth+1;
    asiz  = PetscMax(PetscPowInt(maxS,depth)*coneSeries,PetscPowInt(maxC,depth)*supportSeries);
    asiz *= maxAnchors;
    asiz  = PetscMin(asiz,pEnd-pStart);
    ierr  = PetscMalloc1(asiz,adj);CHKERRQ(ierr);
  }
  if (*adjSize < 0) *adjSize = asiz;
  maxAdjSize = *adjSize;
  if (useTransitiveClosure) {
    ierr = DMPlexGetAdjacency_Transitive_Internal(dm, p, useCone, adjSize, *adj);CHKERRQ(ierr);
  } else if (useCone) {
    ierr = DMPlexGetAdjacency_Cone_Internal(dm, p, adjSize, *adj);CHKERRQ(ierr);
  } else {
    ierr = DMPlexGetAdjacency_Support_Internal(dm, p, adjSize, *adj);CHKERRQ(ierr);
  }
  if (useConstraints && aSec) {
    PetscInt origSize = *adjSize;
    PetscInt numAdj = origSize;
    PetscInt i = 0, j;
    PetscInt *orig = *adj;

    while (i < origSize) {
      PetscInt p = orig[i];
      PetscInt aDof = 0;

      if (p >= aStart && p < aEnd) {
        ierr = PetscSectionGetDof(aSec,p,&aDof);CHKERRQ(ierr);
      }
      if (aDof) {
        PetscInt aOff;
        PetscInt s, q;

        for (j = i + 1; j < numAdj; j++) {
          orig[j - 1] = orig[j];
        }
        origSize--;
        numAdj--;
        ierr = PetscSectionGetOffset(aSec,p,&aOff);CHKERRQ(ierr);
        for (s = 0; s < aDof; ++s) {
          for (q = 0; q < numAdj || (orig[numAdj++] = anchors[aOff+s],0); ++q) {
            if (anchors[aOff+s] == orig[q]) break;
          }
          if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
        }
      }
      else {
        i++;
      }
    }
    *adjSize = numAdj;
    ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency"
/*@
  DMPlexGetAdjacency - Return all points adjacent to the given point

  Input Parameters:
+ dm - The DM object
. p  - The point
. adjSize - The maximum size of adj if it is non-NULL, or PETSC_DETERMINE
- adj - Either NULL so that the array is allocated, or an existing array with size adjSize

  Output Parameters:
+ adjSize - The number of adjacent points
- adj - The adjacent points

  Level: advanced

  Notes: The user must PetscFree the adj array if it was not passed in.

.seealso: DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure(), DMPlexDistribute(), DMCreateMatrix(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexGetAdjacency(DM dm, PetscInt p, PetscInt *adjSize, PetscInt *adj[])
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(adjSize,3);
  PetscValidPointer(adj,4);
  ierr = DMPlexGetAdjacency_Internal(dm, p, mesh->useCone, mesh->useClosure, mesh->useConstraints, adjSize, adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeField"
/*@
  DMPlexDistributeField - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on DM

  Input Parameters:
+ dm - The DMPlex object
. pointSF - The PetscSF describing the communication pattern
. originalSection - The PetscSection for existing data layout
- originalVec - The existing data

  Output Parameters:
+ newSection - The PetscSF describing the new data layout
- newVec - The new data

  Level: developer

.seealso: DMPlexDistribute(), DMPlexDistributeData()
@*/
PetscErrorCode DMPlexDistributeField(DM dm, PetscSF pointSF, PetscSection originalSection, Vec originalVec, PetscSection newSection, Vec newVec)
{
  PetscSF        fieldSF;
  PetscInt      *remoteOffsets, fieldSize;
  PetscScalar   *originalValues, *newValues;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_DistributeField,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection);CHKERRQ(ierr);

  ierr = PetscSectionGetStorageSize(newSection, &fieldSize);CHKERRQ(ierr);
  ierr = VecSetSizes(newVec, fieldSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(newVec,dm->vectype);CHKERRQ(ierr);

  ierr = VecGetArray(originalVec, &originalValues);CHKERRQ(ierr);
  ierr = VecGetArray(newVec, &newValues);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, MPIU_SCALAR, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, MPIU_SCALAR, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  ierr = VecRestoreArray(newVec, &newValues);CHKERRQ(ierr);
  ierr = VecRestoreArray(originalVec, &originalValues);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeField,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeData"
/*@
  DMPlexDistributeData - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on DM

  Input Parameters:
+ dm - The DMPlex object
. pointSF - The PetscSF describing the communication pattern
. originalSection - The PetscSection for existing data layout
. datatype - The type of data
- originalData - The existing data

  Output Parameters:
+ newSection - The PetscSF describing the new data layout
- newData - The new data

  Level: developer

.seealso: DMPlexDistribute(), DMPlexDistributeField()
@*/
PetscErrorCode DMPlexDistributeData(DM dm, PetscSF pointSF, PetscSection originalSection, MPI_Datatype datatype, void *originalData, PetscSection newSection, void **newData)
{
  PetscSF        fieldSF;
  PetscInt      *remoteOffsets, fieldSize;
  PetscMPIInt    dataSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_DistributeData,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection);CHKERRQ(ierr);

  ierr = PetscSectionGetStorageSize(newSection, &fieldSize);CHKERRQ(ierr);
  ierr = MPI_Type_size(datatype, &dataSize);CHKERRQ(ierr);
  ierr = PetscMalloc(fieldSize * dataSize, newData);CHKERRQ(ierr);

  ierr = PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, datatype, originalData, *newData);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, datatype, originalData, *newData);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeData,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistribute"
/*@C
  DMPlexDistribute - Distributes the mesh across processes.

  Collective on DM

  Input Parameters:
+ dm  - The original DMPlex object
. partitioner - The partitioning package, or NULL for the default
- overlap - The overlap of partitions, 0 is the default

  Output Parameters:
+ sf - The PetscSF used for point distribution
- parallelMesh - The distributed DMPlex object, or NULL

  Note: If the mesh was not distributed, the return value is NULL.

  The user can control the definition of adjacency for the mesh using DMPlexGetAdjacencyUseCone() and
  DMPlexSetAdjacencyUseClosure(). They should choose the combination appropriate for the function
  representation on the mesh.

  Level: intermediate

.keywords: mesh, elements
.seealso: DMPlexCreate(), DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure()
@*/
PetscErrorCode DMPlexDistribute(DM dm, const char partitioner[], PetscInt overlap, PetscSF *sf, DM *dmParallel)
{
  DM_Plex               *mesh   = (DM_Plex*) dm->data, *pmesh;
  MPI_Comm               comm;
  const PetscInt         height = 0;
  PetscInt               dim, numRemoteRanks;
  IS                     origCellPart,        origPart,        cellPart,        part;
  PetscSection           origCellPartSection, origPartSection, cellPartSection, partSection;
  PetscSFNode           *remoteRanks;
  PetscSF                partSF, pointSF, coneSF;
  ISLocalToGlobalMapping renumbering;
  PetscSection           originalConeSection, newConeSection;
  PetscInt              *remoteOffsets;
  PetscInt              *cones, *newCones, newConesSize;
  PetscBool              flg;
  PetscMPIInt            rank, numProcs, p;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidPointer(sf,4);
  PetscValidPointer(dmParallel,5);

  ierr = PetscLogEventBegin(DMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);

  *dmParallel = NULL;
  if (numProcs == 1) PetscFunctionReturn(0);

  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create cell partition - We need to rewrite to use IS, use the MatPartition stuff */
  ierr = PetscLogEventBegin(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  if (overlap > 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Overlap > 1 not yet implemented");
  ierr = DMPlexCreatePartition(dm, partitioner, numProcs, height, overlap > 0 ? PETSC_TRUE : PETSC_FALSE, &cellPartSection, &cellPart, &origCellPartSection, &origCellPart);CHKERRQ(ierr);
  /* Create SF assuming a serial partition for all processes: Could check for IS length here */
  if (!rank) numRemoteRanks = numProcs;
  else       numRemoteRanks = 0;
  ierr = PetscMalloc1(numRemoteRanks, &remoteRanks);CHKERRQ(ierr);
  for (p = 0; p < numRemoteRanks; ++p) {
    remoteRanks[p].rank  = p;
    remoteRanks[p].index = 0;
  }
  ierr = PetscSFCreate(comm, &partSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(partSF, 1, numRemoteRanks, NULL, PETSC_OWN_POINTER, remoteRanks, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-partition_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Cell Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(cellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(cellPart, NULL);CHKERRQ(ierr);
    if (origCellPart) {
      ierr = PetscPrintf(comm, "Original Cell Partition:\n");CHKERRQ(ierr);
      ierr = PetscSectionView(origCellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = ISView(origCellPart, NULL);CHKERRQ(ierr);
    }
    ierr = PetscSFView(partSF, NULL);CHKERRQ(ierr);
  }
  /* Close the partition over the mesh */
  ierr = DMPlexCreatePartitionClosure(dm, cellPartSection, cellPart, &partSection, &part);CHKERRQ(ierr);
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  /* Create new mesh */
  ierr  = DMPlexCreate(comm, dmParallel);CHKERRQ(ierr);
  ierr  = DMPlexSetDimension(*dmParallel, dim);CHKERRQ(ierr);
  ierr  = PetscObjectSetName((PetscObject) *dmParallel, "Parallel Mesh");CHKERRQ(ierr);
  pmesh = (DM_Plex*) (*dmParallel)->data;
  /* Distribute sieve points and the global point numbering (replaces creating remote bases) */
  ierr = PetscSFConvertPartition(partSF, partSection, part, &renumbering, &pointSF);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Point Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(partSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(part, NULL);CHKERRQ(ierr);
    ierr = PetscSFView(pointSF, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Point Renumbering after partition:\n");CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(renumbering, NULL);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_DistributeCones,dm,0,0,0);CHKERRQ(ierr);
  /* Distribute cone section */
  ierr = DMPlexGetConeSection(dm, &originalConeSection);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(*dmParallel, &newConeSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalConeSection, &remoteOffsets, newConeSection);CHKERRQ(ierr);
  ierr = DMSetUp(*dmParallel);CHKERRQ(ierr);
  {
    PetscInt pStart, pEnd, p;

    ierr = PetscSectionGetChart(newConeSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt coneSize;
      ierr               = PetscSectionGetDof(newConeSection, p, &coneSize);CHKERRQ(ierr);
      pmesh->maxConeSize = PetscMax(pmesh->maxConeSize, coneSize);
    }
  }
  /* Communicate and renumber cones */
  ierr = PetscSFCreateSectionSF(pointSF, originalConeSection, remoteOffsets, newConeSection, &coneSF);CHKERRQ(ierr);
  ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetCones(*dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newConeSection, &newConesSize);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApplyBlock(renumbering, IS_GTOLM_MASK, newConesSize, newCones, NULL, newCones);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-cones_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Serial Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(originalConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Parallel Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(newConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSFView(coneSF, NULL);CHKERRQ(ierr);
  }
  ierr = DMPlexGetConeOrientations(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientations(*dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&coneSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeCones,dm,0,0,0);CHKERRQ(ierr);
  /* Create supports and stratify sieve */
  {
    PetscInt pStart, pEnd;

    ierr = PetscSectionGetChart(pmesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(pmesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(*dmParallel);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dmParallel);CHKERRQ(ierr);
  /* Create point SF for parallel mesh */
  ierr = PetscLogEventBegin(DMPLEX_DistributeSF,dm,0,0,0);CHKERRQ(ierr);
  {
    const PetscInt *leaves;
    PetscSFNode    *remotePoints, *rowners, *lowners;
    PetscInt        numRoots, numLeaves, numGhostPoints = 0, p, gp, *ghostPoints;
    PetscInt        pStart, pEnd;

    ierr = DMPlexGetChart(*dmParallel, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(pointSF, &numRoots, &numLeaves, &leaves, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc2(numRoots,&rowners,numLeaves,&lowners);CHKERRQ(ierr);
    for (p=0; p<numRoots; p++) {
      rowners[p].rank  = -1;
      rowners[p].index = -1;
    }
    if (origCellPart) {
      /* Make sure points in the original partition are not assigned to other procs */
      const PetscInt *origPoints;

      ierr = DMPlexCreatePartitionClosure(dm, origCellPartSection, origCellPart, &origPartSection, &origPart);CHKERRQ(ierr);
      ierr = ISGetIndices(origPart, &origPoints);CHKERRQ(ierr);
      for (p = 0; p < numProcs; ++p) {
        PetscInt dof, off, d;

        ierr = PetscSectionGetDof(origPartSection, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(origPartSection, p, &off);CHKERRQ(ierr);
        for (d = off; d < off+dof; ++d) {
          rowners[origPoints[d]].rank = p;
        }
      }
      ierr = ISRestoreIndices(origPart, &origPoints);CHKERRQ(ierr);
      ierr = ISDestroy(&origPart);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&origPartSection);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&origCellPart);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&origCellPartSection);CHKERRQ(ierr);

    ierr = PetscSFBcastBegin(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      if (lowners[p].rank < 0 || lowners[p].rank == rank) { /* Either put in a bid or we know we own it */
        lowners[p].rank  = rank;
        lowners[p].index = leaves ? leaves[p] : p;
      } else if (lowners[p].rank >= 0) { /* Point already claimed so flag so that MAXLOC does not listen to us */
        lowners[p].rank  = -2;
        lowners[p].index = -2;
      }
    }
    for (p=0; p<numRoots; p++) { /* Root must not participate in the rediction, flag so that MAXLOC does not use */
      rowners[p].rank  = -3;
      rowners[p].index = -3;
    }
    ierr = PetscSFReduceBegin(pointSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(pointSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      if (lowners[p].rank < 0 || lowners[p].index < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cell partition corrupt: point not claimed");
      if (lowners[p].rank != rank) ++numGhostPoints;
    }
    ierr = PetscMalloc1(numGhostPoints,    &ghostPoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(numGhostPoints, &remotePoints);CHKERRQ(ierr);
    for (p = 0, gp = 0; p < numLeaves; ++p) {
      if (lowners[p].rank != rank) {
        ghostPoints[gp]        = leaves ? leaves[p] : p;
        remotePoints[gp].rank  = lowners[p].rank;
        remotePoints[gp].index = lowners[p].index;
        ++gp;
      }
    }
    ierr = PetscFree2(rowners,lowners);CHKERRQ(ierr);
    ierr = PetscSFSetGraph((*dmParallel)->sf, pEnd - pStart, numGhostPoints, ghostPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions((*dmParallel)->sf);CHKERRQ(ierr);
  }
  pmesh->useCone    = mesh->useCone;
  pmesh->useClosure = mesh->useClosure;
  ierr = PetscLogEventEnd(DMPLEX_DistributeSF,dm,0,0,0);CHKERRQ(ierr);
  /* Distribute Coordinates */
  {
    PetscSection     originalCoordSection, newCoordSection;
    Vec              originalCoordinates, newCoordinates;
    PetscInt         bs;
    const char      *name;
    const PetscReal *maxCell, *L;

    ierr = DMGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(*dmParallel, &newCoordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &originalCoordinates);CHKERRQ(ierr);
    ierr = VecCreate(comm, &newCoordinates);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) originalCoordinates, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) newCoordinates, name);CHKERRQ(ierr);

    ierr = DMPlexDistributeField(dm, pointSF, originalCoordSection, originalCoordinates, newCoordSection, newCoordinates);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*dmParallel, newCoordinates);CHKERRQ(ierr);
    ierr = VecGetBlockSize(originalCoordinates, &bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(newCoordinates, bs);CHKERRQ(ierr);
    ierr = VecDestroy(&newCoordinates);CHKERRQ(ierr);
    ierr = DMGetPeriodicity(dm, &maxCell, &L);CHKERRQ(ierr);
    if (L) {ierr = DMSetPeriodicity(*dmParallel, maxCell, L);CHKERRQ(ierr);}
  }
  /* Distribute labels */
  ierr = PetscLogEventBegin(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  {
    DMLabel  next      = mesh->labels, newNext = pmesh->labels;
    PetscInt numLabels = 0, l;

    /* Bcast number of labels */
    while (next) {++numLabels; next = next->next;}
    ierr = MPI_Bcast(&numLabels, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
    next = mesh->labels;
    for (l = 0; l < numLabels; ++l) {
      DMLabel   labelNew;
      PetscBool isdepth;

      /* Skip "depth" because it is recreated */
      if (!rank) {ierr = PetscStrcmp(next->name, "depth", &isdepth);CHKERRQ(ierr);}
      ierr = MPI_Bcast(&isdepth, 1, MPIU_BOOL, 0, comm);CHKERRQ(ierr);
      if (isdepth) {if (!rank) next = next->next; continue;}
      ierr = DMLabelDistribute(next, partSection, part, renumbering, &labelNew);CHKERRQ(ierr);
      /* Insert into list */
      if (newNext) newNext->next = labelNew;
      else         pmesh->labels = labelNew;
      newNext = labelNew;
      if (!rank) next = next->next;
    }
  }
  ierr = PetscLogEventEnd(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  /* Setup hybrid structure */
  {
    const PetscInt *gpoints;
    PetscInt        depth, n, d;

    for (d = 0; d <= dim; ++d) {pmesh->hybridPointMax[d] = mesh->hybridPointMax[d];}
    ierr = MPI_Bcast(pmesh->hybridPointMax, dim+1, MPIU_INT, 0, comm);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(renumbering, &n);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(renumbering, &gpoints);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    for (d = 0; d <= dim; ++d) {
      PetscInt pmax = pmesh->hybridPointMax[d], newmax = 0, pEnd, stratum[2], p;

      if (pmax < 0) continue;
      ierr = DMPlexGetDepthStratum(dm, d > depth ? depth : d, &stratum[0], &stratum[1]);CHKERRQ(ierr);
      ierr = DMPlexGetDepthStratum(*dmParallel, d, NULL, &pEnd);CHKERRQ(ierr);
      ierr = MPI_Bcast(stratum, 2, MPIU_INT, 0, comm);CHKERRQ(ierr);
      for (p = 0; p < n; ++p) {
        const PetscInt point = gpoints[p];

        if ((point >= stratum[0]) && (point < stratum[1]) && (point >= pmax)) ++newmax;
      }
      if (newmax > 0) pmesh->hybridPointMax[d] = pEnd - newmax;
      else            pmesh->hybridPointMax[d] = -1;
    }
    ierr = ISLocalToGlobalMappingRestoreIndices(renumbering, &gpoints);CHKERRQ(ierr);
  }
  /* Cleanup Partition */
  ierr = ISLocalToGlobalMappingDestroy(&renumbering);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&partSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  ierr = ISDestroy(&part);CHKERRQ(ierr);
  /* Copy BC */
  ierr = DMPlexCopyBoundary(dm, *dmParallel);CHKERRQ(ierr);
  /* Cleanup */
  if (sf) {*sf = pointSF;}
  else    {ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);}
  ierr = DMSetFromOptions(*dmParallel);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDecompose"
/*@C
  DMPlexDecompose - Decomposes the mesh into many regions per process.

  Collective on DM

  Input Parameters:
+ dm - The original DMPlex object
. partitioner - The partitioning package, or NULL for the default
. n - The number of partitions
- overlap - The overlap of partitions, 0 is the default

  Output Parameters:
+ origPartSection - Offsets into 'part' for partitions without overlap
. origPart - Array of partition points
. partitionSection - Offsets into 'part' for partitions with overlap
. partition - Array of partition points
- dms - The individual DMPlex objects for each domain

  The user can control the definition of adjacency for the mesh using DMPlexGetAdjacencyUseCone() and
  DMPlexSetAdjacencyUseClosure(). They should choose the combination appropriate for the function
  representation on the mesh.

  Level: intermediate

.keywords: mesh, elements
.seealso: DMPlexDistribute(), DMPlexCreate(), DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure()
@*/
PetscErrorCode DMPlexDecompose(DM dm, const char partitioner[], PetscInt n, PetscInt overlap, PetscSection *origPartitionSection, IS *origPartition, PetscSection *partitionSection, IS *partition, DM **dms)
{
  DM_Plex               *mesh   = (DM_Plex *) dm->data, *pmesh;
  const PetscInt         height = 0;
  MPI_Comm               comm;
  PetscInt               dim, numRemoteRanks;
  IS                     origCellPart,        cellPart,        origPart,        part;
  PetscSection           origCellPartSection, cellPartSection, origPartSection, partSection;
  PetscSFNode           *remoteRanks;
  PetscSF                partSF, pointSF, coneSF;
  ISLocalToGlobalMapping renumbering;
  PetscSection           originalConeSection, newConeSection;
  PetscInt              *remoteOffsets;
  PetscInt              *cones, *newCones, newConesSize;
  PetscBool              flg;
  PetscMPIInt            rank, numProcs, p, i;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dms,5);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  if (n < 1) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Must have at least one partition, not %d", n);
  ierr = PetscLogEventBegin(DMPLEX_Decompose,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, dms);CHKERRQ(ierr);
  if (n == 1) {
    (*dms)[0] = dm;
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);

  /* Create cell partition - We need to rewrite to use IS, use the MatPartition stuff */
  if (overlap > 1) SETERRQ(comm, PETSC_ERR_SUP, "Overlap > 1 not yet implemented");
  ierr = PetscLogEventBegin(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexCreatePartition(dm, partitioner, n, height, overlap > 0 ? PETSC_TRUE : PETSC_FALSE, &cellPartSection, &cellPart, &origCellPartSection, &origCellPart);CHKERRQ(ierr);
  /* Create SF assuming partitions are local: Could use current SF info here */
  numRemoteRanks = 1;
  ierr = PetscMalloc1(numRemoteRanks, &remoteRanks);CHKERRQ(ierr);
  for (p = 0; p < numRemoteRanks; ++p) {
    remoteRanks[p].rank  = rank;
    remoteRanks[p].index = 0;
  }
  ierr = PetscSFCreate(comm, &partSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(partSF, 1, numRemoteRanks, NULL, PETSC_OWN_POINTER, remoteRanks, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-partition_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Cell Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(cellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(cellPart, NULL);CHKERRQ(ierr);
    if (origCellPart) {
      ierr = PetscPrintf(comm, "Original Cell Partition:\n");CHKERRQ(ierr);
      ierr = PetscSectionView(origCellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = ISView(origCellPart, NULL);CHKERRQ(ierr);
    }
    ierr = PetscSFView(partSF, NULL);CHKERRQ(ierr);
  }
  /* Close the partition over the mesh */
  ierr = DMPlexCreatePartitionClosure(dm, origCellPartSection, origCellPart, &origPartSection, &origPart);CHKERRQ(ierr);
  ierr = ISDestroy(&origCellPart);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&origCellPartSection);CHKERRQ(ierr);
  ierr = DMPlexCreatePartitionClosure(dm, cellPartSection, cellPart, &partSection, &part);CHKERRQ(ierr);
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Point Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(partSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(part, NULL);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  /* Create new meshes */
  for (i = 0; i < n; ++i) {
    DM   pdm;
    char name[256];

    ierr  = DMPlexCreate(comm, &pdm);CHKERRQ(ierr);
    ierr  = DMPlexSetDimension(pdm, dim);CHKERRQ(ierr);
    ierr  = PetscSNPrintf(name, 255, "Subdomain %D", i);CHKERRQ(ierr);
    ierr  = PetscObjectSetName((PetscObject) pdm, name);CHKERRQ(ierr);
    pmesh = (DM_Plex*) (pdm)->data;
    (*dms)[i] = pdm;
    /* Assume a serial partition, so no pointSF but we will need renumbering I think */
    ierr = PetscLogEventBegin(DMPLEX_DistributeCones,dm,0,0,0);CHKERRQ(ierr);
    {
      const PetscInt *points;
      PetscInt       *leaves;
      PetscSFNode    *remotes;
      PetscInt        numPoints, partSize, partOff;

      ierr = PetscSectionGetDof(partSection, i, &partSize);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(partSection, i, &partOff);CHKERRQ(ierr);
      ierr = PetscMalloc1(partSize,&leaves);CHKERRQ(ierr);
      ierr = PetscMalloc1(partSize,&remotes);CHKERRQ(ierr);
      ierr = ISGetLocalSize(part, &numPoints);CHKERRQ(ierr);
      ierr = ISGetIndices(part, &points);CHKERRQ(ierr);
      for (p = 0; p < partSize; ++p) {
        leaves[p]        = p;
        remotes[p].rank  = rank;
        remotes[p].index = points[partOff+p];
      }
      ierr = ISRestoreIndices(part, &points);CHKERRQ(ierr);
      ierr = PetscSFCreate(comm, &pointSF);CHKERRQ(ierr);
      ierr = PetscSFSetFromOptions(pointSF);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(pointSF, numPoints, partSize, leaves, PETSC_OWN_POINTER, remotes, PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreateSF(pointSF, 0, &renumbering);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscSFView(pointSF, NULL);CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "Point Renumbering for partition %D:\n", i);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingView(renumbering, NULL);CHKERRQ(ierr);
      }
    }
    /* Distribute cone section */
    ierr = DMPlexGetConeSection(dm, &originalConeSection);CHKERRQ(ierr);
    ierr = DMPlexGetConeSection(pdm, &newConeSection);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(pointSF, originalConeSection, &remoteOffsets, newConeSection);CHKERRQ(ierr);
    ierr = DMSetUp(pdm);CHKERRQ(ierr);
    {
      PetscInt pStart, pEnd, p;

      ierr = PetscSectionGetChart(newConeSection, &pStart, &pEnd);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        PetscInt coneSize;
        ierr               = PetscSectionGetDof(newConeSection, p, &coneSize);CHKERRQ(ierr);
        pmesh->maxConeSize = PetscMax(pmesh->maxConeSize, coneSize);
      }
    }
    /* Communicate and renumber cones */
    ierr = PetscSFCreateSectionSF(pointSF, originalConeSection, remoteOffsets, newConeSection, &coneSF);CHKERRQ(ierr);
    ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
    ierr = DMPlexGetCones(pdm, &newCones);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(newConeSection, &newConesSize);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApplyBlock(renumbering, IS_GTOLM_MASK, newConesSize, newCones, NULL, newCones);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-cones_view", &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscPrintf(comm, "Serial Cone Section:\n");CHKERRQ(ierr);
      ierr = PetscSectionView(originalConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "Parallel Cone Section:\n");CHKERRQ(ierr);
      ierr = PetscSectionView(newConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = PetscSFView(coneSF, NULL);CHKERRQ(ierr);
    }
    ierr = DMPlexGetConeOrientations(dm, &cones);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientations(pdm, &newCones);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&coneSF);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(DMPLEX_DistributeCones,dm,0,0,0);CHKERRQ(ierr);
    /* Create supports and stratify sieve */
    {
      PetscInt pStart, pEnd;

      ierr = PetscSectionGetChart(pmesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(pmesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
    }
    ierr = DMPlexSymmetrize(pdm);CHKERRQ(ierr);
    ierr = DMPlexStratify(pdm);CHKERRQ(ierr);
    /* Assume serial partitions, so no point SF */
    /* Distribute Coordinates */
    {
      PetscSection originalCoordSection, newCoordSection;
      Vec          originalCoordinates, newCoordinates;
      PetscInt     bs;
      const char  *name;

      ierr = DMGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
      ierr = DMGetCoordinateSection(pdm, &newCoordSection);CHKERRQ(ierr);
      ierr = DMGetCoordinatesLocal(dm, &originalCoordinates);CHKERRQ(ierr);
      ierr = VecCreate(comm, &newCoordinates);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) originalCoordinates, &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) newCoordinates, name);CHKERRQ(ierr);

      ierr = DMPlexDistributeField(dm, pointSF, originalCoordSection, originalCoordinates, newCoordSection, newCoordinates);CHKERRQ(ierr);
      ierr = DMSetCoordinatesLocal(pdm, newCoordinates);CHKERRQ(ierr);
      ierr = VecGetBlockSize(originalCoordinates, &bs);CHKERRQ(ierr);
      ierr = VecSetBlockSize(newCoordinates, bs);CHKERRQ(ierr);
      ierr = VecDestroy(&newCoordinates);CHKERRQ(ierr);
    }
    /* Distribute labels */
    ierr = PetscLogEventBegin(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
    {
      DMLabel next = mesh->labels, newNext = pmesh->labels;

      while (next) {
        DMLabel   labelNew;
        PetscBool isdepth;

        /* Skip "depth" because it is recreated */
        ierr = PetscStrcmp(next->name, "depth", &isdepth);CHKERRQ(ierr);
        if (isdepth) {next = next->next; continue;}
        ierr = DMLabelDecompose(next, partSection, part, i, renumbering, &labelNew);CHKERRQ(ierr);
        /* Insert into list */
        if (newNext) newNext->next = labelNew;
        else         pmesh->labels = labelNew;
        newNext = labelNew;
        next = next->next;
      }
    }
    ierr = PetscLogEventEnd(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
    /* Setup hybrid structure */
    {
      const PetscInt *gpoints;
      PetscInt        depth, n, d;

      for (d = 0; d <= dim; ++d) {pmesh->hybridPointMax[d] = mesh->hybridPointMax[d];}
      ierr = MPI_Bcast(pmesh->hybridPointMax, dim+1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetSize(renumbering, &n);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetIndices(renumbering, &gpoints);CHKERRQ(ierr);
      ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
      for (d = 0; d <= dim; ++d) {
        PetscInt pmax = pmesh->hybridPointMax[d], newmax = 0, pEnd, stratum[2], p;

        if (pmax < 0) continue;
        ierr = DMPlexGetDepthStratum(dm, d > depth ? depth : d, &stratum[0], &stratum[1]);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(pdm, d, NULL, &pEnd);CHKERRQ(ierr);
        ierr = MPI_Bcast(stratum, 2, MPIU_INT, 0, comm);CHKERRQ(ierr);
        for (p = 0; p < n; ++p) {
          const PetscInt point = gpoints[p];

          if ((point >= stratum[0]) && (point < stratum[1]) && (point >= pmax)) ++newmax;
        }
        if (newmax > 0) pmesh->hybridPointMax[d] = pEnd - newmax;
        else            pmesh->hybridPointMax[d] = -1;
      }
      ierr = ISLocalToGlobalMappingRestoreIndices(renumbering, &gpoints);CHKERRQ(ierr);
    }
    /* Cleanup Partition */
    ierr = ISLocalToGlobalMappingDestroy(&renumbering);CHKERRQ(ierr);
    /* Copy BC */
    ierr = DMPlexCopyBoundary(dm, pdm);CHKERRQ(ierr);
    /* Copy DS */
    {
      PetscDS prob;

      ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
      ierr = DMSetDS(pdm, prob);CHKERRQ(ierr);
    }
    /* Cleanup */
    ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);
    ierr = DMSetFromOptions(pdm);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&partSF);CHKERRQ(ierr);
  if (origPartition) {
    *origPartitionSection = origPartSection;
    *origPartition        = origPart;
  } else {
    ierr = PetscSectionDestroy(&origPartSection);CHKERRQ(ierr);
    ierr = ISDestroy(&origPart);CHKERRQ(ierr);
  }
  if (partition) {
    *partitionSection = partSection;
    *partition        = part;
  } else {
    ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
    ierr = ISDestroy(&part);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_Decompose,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
