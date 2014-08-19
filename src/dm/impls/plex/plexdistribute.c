#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

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
PetscErrorCode DMPlexGetAdjacency_Internal(DM dm, PetscInt p, PetscBool useCone, PetscBool useTransitiveClosure, PetscInt *adjSize, PetscInt *adj[])
{
  static PetscInt asiz = 0;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  if (!*adj) {
    PetscInt depth, maxConeSize, maxSupportSize;

    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    asiz = PetscPowInt(maxConeSize, depth+1) * PetscPowInt(maxSupportSize, depth+1) + 1;
    ierr = PetscMalloc1(asiz,adj);CHKERRQ(ierr);
  }
  if (*adjSize < 0) *adjSize = asiz;
  if (useTransitiveClosure) {
    ierr = DMPlexGetAdjacency_Transitive_Internal(dm, p, useCone, adjSize, *adj);CHKERRQ(ierr);
  } else if (useCone) {
    ierr = DMPlexGetAdjacency_Cone_Internal(dm, p, adjSize, *adj);CHKERRQ(ierr);
  } else {
    ierr = DMPlexGetAdjacency_Support_Internal(dm, p, adjSize, *adj);CHKERRQ(ierr);
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
  ierr = DMPlexGetAdjacency_Internal(dm, p, mesh->useCone, mesh->useClosure, adjSize, adj);CHKERRQ(ierr);
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
#define __FUNCT__ "DMPlexDistributeCones"
PetscErrorCode DMPlexDistributeCones(DM dm, PetscSF migrationSF, ISLocalToGlobalMapping renumbering, DM *dmParallel)
{
  DM_Plex               *pmesh   = (DM_Plex*) (*dmParallel)->data;
  MPI_Comm               comm;
  PetscSF                coneSF;
  PetscSection           originalConeSection, newConeSection;
  PetscInt              *remoteOffsets, *cones, *newCones, newConesSize;
  PetscBool              flg;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmParallel,4);
  ierr = PetscLogEventBegin(DMPLEX_DistributeCones,dm,0,0,0);CHKERRQ(ierr);

  /* Distribute cone section */
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(dm, &originalConeSection);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(*dmParallel, &newConeSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(migrationSF, originalConeSection, &remoteOffsets, newConeSection);CHKERRQ(ierr);
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
  ierr = PetscSFCreateSectionSF(migrationSF, originalConeSection, remoteOffsets, newConeSection, &coneSF);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeCoordinates"
PetscErrorCode DMPlexDistributeCoordinates(DM dm, PetscSF migrationSF, DM *dmParallel)
{
  MPI_Comm         comm;
  PetscSection     originalCoordSection, newCoordSection;
  Vec              originalCoordinates, newCoordinates;
  PetscInt         bs;
  const char      *name;
  const PetscReal *maxCell, *L;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmParallel, 3);

  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(*dmParallel, &newCoordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &originalCoordinates);CHKERRQ(ierr);
  ierr = VecCreate(comm, &newCoordinates);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) originalCoordinates, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newCoordinates, name);CHKERRQ(ierr);

  ierr = DMPlexDistributeField(dm, migrationSF, originalCoordSection, originalCoordinates, newCoordSection, newCoordinates);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dmParallel, newCoordinates);CHKERRQ(ierr);
  ierr = VecGetBlockSize(originalCoordinates, &bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(newCoordinates, bs);CHKERRQ(ierr);
  ierr = VecDestroy(&newCoordinates);CHKERRQ(ierr);
  ierr = DMGetPeriodicity(dm, &maxCell, &L);CHKERRQ(ierr);
  if (L) {ierr = DMSetPeriodicity(*dmParallel, maxCell, L);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeLabels"
PetscErrorCode DMPlexDistributeLabels(DM dm, PetscSF migrationSF, DM *dmParallel)
{
  DM_Plex       *mesh      = (DM_Plex*) dm->data;
  DM_Plex       *pmesh     = (DM_Plex*) (*dmParallel)->data;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DMLabel        next      = mesh->labels, newNext = pmesh->labels;
  PetscInt       numLabels = 0, l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmParallel, 6);
  ierr = PetscLogEventBegin(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

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
    if (isdepth) {if(next) next = next->next; continue;}
    ierr = DMLabelDistribute(next, migrationSF, &labelNew);CHKERRQ(ierr);
    /* Insert into list */
    if (newNext) newNext->next = labelNew;
    else         pmesh->labels = labelNew;
    newNext = labelNew;
    if (next) next = next->next;
  }
  ierr = PetscLogEventEnd(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeSetupHybrid"
PetscErrorCode DMPlexDistributeSetupHybrid(DM dm, PetscSF migrationSF, ISLocalToGlobalMapping renumbering, DM *dmParallel)
{
  DM_Plex        *mesh  = (DM_Plex*) dm->data;
  DM_Plex        *pmesh = (DM_Plex*) (*dmParallel)->data;
  MPI_Comm        comm;
  const PetscInt *gpoints;
  PetscInt        dim, depth, n, d;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmParallel, 4);

  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  /* Setup hybrid structure */
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
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeSF"
PetscErrorCode DMPlexDistributeSF(DM dm, PetscSF migrationSF, PetscSection partSection, IS part, PetscSection origPartSection, IS origPart, DM *dmParallel)
{
  DM_Plex               *mesh  = (DM_Plex*) dm->data;
  DM_Plex               *pmesh = (DM_Plex*) (*dmParallel)->data;
  PetscMPIInt            rank, numProcs;
  MPI_Comm               comm;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmParallel,7);

  /* Create point SF for parallel mesh */
  ierr = PetscLogEventBegin(DMPLEX_DistributeSF,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  {
    const PetscInt *leaves;
    PetscSFNode    *remotePoints, *rowners, *lowners;
    PetscInt        numRoots, numLeaves, numGhostPoints = 0, p, gp, *ghostPoints;
    PetscInt        pStart, pEnd;

    ierr = DMPlexGetChart(*dmParallel, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(migrationSF, &numRoots, &numLeaves, &leaves, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc2(numRoots,&rowners,numLeaves,&lowners);CHKERRQ(ierr);
    for (p=0; p<numRoots; p++) {
      rowners[p].rank  = -1;
      rowners[p].index = -1;
    }
    if (origPart) {
      /* Make sure points in the original partition are not assigned to other procs */
      const PetscInt *origPoints;

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
    }
    ierr = PetscSFBcastBegin(migrationSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(migrationSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
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
    ierr = PetscSFReduceBegin(migrationSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(migrationSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(migrationSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(migrationSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistribute"
/*@C
  DMPlexDistribute - Distributes the mesh and any associated sections.

  Not Collective

  Input Parameter:
+ dm  - The original DMPlex object
. partitioner - The partitioning package, or NULL for the default
- overlap - The overlap of partitions, 0 is the default

  Output Parameter:
+ sf - The PetscSF used for point distribution
- parallelMesh - The distributed DMPlex object, or NULL

  Note: If the mesh was not distributed, the return value is NULL.

  The user can control the definition of adjacency for the mesh using DMPlexGetAdjacencyUseCone() and
  DMPlexSetAdjacencyUseClosure(). They should choose the combination appropriate for the function
  representation on the mesh.

  Level: intermediate

.keywords: mesh, elements
.seealso: DMPlexCreate(), DMPlexDistributeByFace(), DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure()
@*/
PetscErrorCode DMPlexDistribute(DM dm, const char partitioner[], PetscInt overlap, PetscSF *sf, DM *dmParallel)
{
  MPI_Comm               comm;
  const PetscInt         height = 0;
  PetscInt               dim, numRemoteRanks;
  DM                     dmOverlap;
  IS                     cellPart,        part;
  PetscSection           cellPartSection, partSection;
  PetscSFNode           *remoteRanks;
  PetscSF                partSF, pointSF, overlapSF;
  ISLocalToGlobalMapping renumbering;
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

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create cell partition - We need to rewrite to use IS, use the MatPartition stuff */
  ierr = PetscLogEventBegin(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  if (overlap > 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Overlap > 1 not yet implemented");
  ierr = DMPlexCreatePartition(dm, partitioner, height, PETSC_FALSE, &cellPartSection, &cellPart, NULL, NULL);CHKERRQ(ierr);
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
    ierr = PetscPrintf(comm, "Original Cell Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(cellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(cellPart, NULL);CHKERRQ(ierr);
    ierr = PetscSFView(partSF, NULL);CHKERRQ(ierr);
  }
  /* Close the partition over the mesh */
  ierr = DMPlexCreatePartitionClosure(dm, cellPartSection, cellPart, &partSection, &part);CHKERRQ(ierr);
  /* Create new mesh */
  ierr  = DMPlexCreate(comm, dmParallel);CHKERRQ(ierr);
  ierr  = DMSetDimension(*dmParallel, dim);CHKERRQ(ierr);
  ierr  = PetscObjectSetName((PetscObject) *dmParallel, "Parallel Mesh");CHKERRQ(ierr);
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

  /* Migrate data to a non-overlapping parallel DM */
  ierr = DMPlexDistributeCones(dm, pointSF, renumbering, dmParallel);CHKERRQ(ierr);
  ierr = DMPlexDistributeCoordinates(dm, pointSF, dmParallel);CHKERRQ(ierr);
  ierr = DMPlexDistributeLabels(dm, pointSF, dmParallel);CHKERRQ(ierr);
  ierr = DMPlexDistributeSetupHybrid(dm, pointSF, renumbering, dmParallel);CHKERRQ(ierr);

  /* Build the point SF without overlap */
  ierr = DMPlexDistributeSF(dm, pointSF, partSection, part, NULL, NULL, dmParallel);CHKERRQ(ierr);

  if (overlap > 0) {
    /* Add the partition overlap to the distributed DM */
    ierr = DMPlexDistributeOverlap(*dmParallel, overlap, renumbering, &overlapSF, &dmOverlap);CHKERRQ(ierr);
    ierr = DMDestroy(dmParallel);CHKERRQ(ierr);
    *dmParallel = dmOverlap;
  }
  /* Cleanup Partition */
  ierr = ISLocalToGlobalMappingDestroy(&renumbering);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&partSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  ierr = ISDestroy(&part);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
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
#define __FUNCT__ "DMPlexDistributeOverlap"
/*@C
  DMPlexDistribute - Add partition overlap to a distributed non-overlapping DM.

  Not Collective

  Input Parameter:
+ dm  - The non-overlapping distrbuted DMPlex object
- overlap - The overlap of partitions, 0 is the default

  Output Parameter:
+ sf - The PetscSF used for point distribution
- dmOverlap - The overlapping distributed DMPlex object, or NULL

  Note: If the mesh was not distributed, the return value is NULL.

  The user can control the definition of adjacency for the mesh using DMPlexGetAdjacencyUseCone() and
  DMPlexSetAdjacencyUseClosure(). They should choose the combination appropriate for the function
  representation on the mesh.

  Level: intermediate

.keywords: mesh, elements
.seealso: DMPlexCreate(), DMPlexDistributeByFace(), DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure()
@*/
PetscErrorCode DMPlexDistributeOverlap(DM dm, PetscInt overlap, ISLocalToGlobalMapping renumbering, PetscSF *sf, DM *dmOverlap)
{
  MPI_Comm               comm;
  PetscMPIInt            rank;
  IS                     overlapPartition;
  PetscSection           overlapSection, coneSection;
  PetscSF                overlapSF, migrationSF, pointSF, newPointSF;
  PetscSFNode           *ghostRemote;
  const PetscSFNode     *overlapRemote;
  ISLocalToGlobalMapping overlapRenumbering;
  const PetscInt        *renumberingArray, *overlapLocal;
  PetscInt               dim, p, pStart, pEnd, conesSize, idx;
  PetscInt               numGhostPoints, numOverlapPoints, numSharedPoints, overlapLeaves;
  PetscInt              *cones, *ghostLocal, *overlapRenumberingArray, *pointIDs, *recvPointIDs;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidPointer(sf, 3);
  PetscValidPointer(dmOverlap, 4);

  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  /* Compute point overlap with neighbouring processes on the distributed DM */
  ierr = PetscLogEventBegin(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexCreatePartitionOverlap(dm, &overlapSection, &overlapPartition);CHKERRQ(ierr);
  ierr = DMPlexCreatePartitionSF(dm, overlapSection, overlapPartition, &overlapSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);

  /* Build dense migration SF that maps the non-overlapping partition to the overlapping one */
  ierr = DMPlexCreateOverlapMigrationSF(dm, overlapSF, &migrationSF);CHKERRQ(ierr);

  /* Convert cones to global numbering before migrating them */
  ierr = DMPlexGetConeSection(dm, &coneSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coneSection, &conesSize);CHKERRQ(ierr);
  ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyBlock(renumbering, conesSize, cones, cones);CHKERRQ(ierr);

  /* Derive the new local-to-global mapping from the old one */
  ierr = PetscSFGetGraph(migrationSF, NULL, &overlapLeaves, &overlapLocal, &overlapRemote);CHKERRQ(ierr);
  ierr = PetscMalloc1(overlapLeaves, &overlapRenumberingArray);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(renumbering, &renumberingArray);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(migrationSF, MPIU_INT, renumberingArray, overlapRenumberingArray);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(migrationSF, MPIU_INT, renumberingArray, overlapRenumberingArray);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm, 1, overlapLeaves, (const PetscInt*) overlapRenumberingArray, PETSC_OWN_POINTER, &overlapRenumbering);CHKERRQ(ierr);

  /* Build the overlapping DM */
  ierr = DMPlexCreate(comm, dmOverlap);CHKERRQ(ierr);
  ierr = DMSetDimension(*dmOverlap, dim);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dmOverlap, "Parallel Mesh");CHKERRQ(ierr);
  ierr = DMPlexDistributeCones(dm, migrationSF, overlapRenumbering, dmOverlap);CHKERRQ(ierr);
  ierr = DMPlexDistributeCoordinates(dm, migrationSF, dmOverlap);CHKERRQ(ierr);
  ierr = DMPlexDistributeLabels(dm, migrationSF, dmOverlap);CHKERRQ(ierr);
  ierr = DMPlexDistributeSetupHybrid(dm, migrationSF, overlapRenumbering, dmOverlap);CHKERRQ(ierr);

  /* Build the new point SF by propagating the depthShift generate remote root indices */
  ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(pointSF, NULL, &numSharedPoints, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(overlapSF, NULL, &numOverlapPoints, NULL, NULL);CHKERRQ(ierr);
  numGhostPoints = numSharedPoints + numOverlapPoints;
  ierr = PetscMalloc2(numGhostPoints, &ghostLocal, numGhostPoints, &ghostRemote);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc2(pEnd-pStart, &pointIDs, overlapLeaves, &recvPointIDs);CHKERRQ(ierr);
  for (p=0; p<overlapLeaves; p++) {
    if (overlapRemote[p].rank == rank) pointIDs[overlapRemote[p].index] = overlapLocal[p];
  }
  ierr = PetscSFBcastBegin(migrationSF, MPIU_INT, pointIDs, recvPointIDs);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(migrationSF, MPIU_INT, pointIDs, recvPointIDs);CHKERRQ(ierr);
  for (idx=0, p=0; p<overlapLeaves; p++) {
    if (overlapRemote[p].rank != rank) {
      ghostLocal[idx] = overlapLocal[p];
      ghostRemote[idx].index = recvPointIDs[p];
      ghostRemote[idx].rank = overlapRemote[p].rank;
      idx++;
    }
  }
  ierr = DMPlexGetChart(*dmOverlap, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, &newPointSF);;CHKERRQ(ierr);
  ierr = PetscSFSetGraph(newPointSF, pEnd - pStart, numGhostPoints, ghostLocal, PETSC_OWN_POINTER, ghostRemote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = DMSetPointSF(*dmOverlap, newPointSF);CHKERRQ(ierr);
  /* Cleanup overlap partition */
  ierr = ISLocalToGlobalMappingDestroy(&overlapRenumbering);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&overlapSection);CHKERRQ(ierr);
  ierr = ISDestroy(&overlapPartition);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&overlapSF);CHKERRQ(ierr);
  ierr = PetscFree2(pointIDs, recvPointIDs);CHKERRQ(ierr);
  if (sf) *sf = migrationSF;
  else    {ierr = PetscSFDestroy(&migrationSF);CHKERRQ(ierr);}
  ierr = DMSetFromOptions(*dmOverlap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
