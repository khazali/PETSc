#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <../src/sys/utils/hash.h>
#include <petsc-private/isimpl.h>
#include <petscsf.h>

/** hierarchy routines */

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetReferenceTree"
/*@
  DMPlexSetReferenceTree - set the reference tree for hierarchically non-conforming meshes.

  Not collective

  Input Parameters:
+ dm - The DMPlex object
- ref - The reference tree DMPlex object

  Level: beginner

.seealso: DMPlexGetReferenceTree(), DMPlexCreateDefaultReferenceTree()
@*/
PetscErrorCode DMPlexSetReferenceTree(DM dm, DM ref)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(ref, DM_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject)ref);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh->referenceTree);CHKERRQ(ierr);
  mesh->referenceTree = ref;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetReferenceTree"
/*@
  DMPlexGetReferenceTree - get the reference tree for hierarchically non-conforming meshes.

  Not collective

  Input Parameters:
. dm - The DMPlex object

  Output Parameters
. ref - The reference tree DMPlex object

  Level: beginner

.seealso: DMPlexSetReferenceTree(), DMPlexCreateDefaultReferenceTree()
@*/
PetscErrorCode DMPlexGetReferenceTree(DM dm, DM *ref)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ref,2);
  *ref = mesh->referenceTree;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateDefaultReferenceTree"
/*@
  DMPlexCreateDefaultReferenceTree - create a reference tree for isotropic hierarchical mesh refinement.

  Collective on comm

  Input Parameters:
+ comm    - the MPI communicator
. dim     - the spatial dimension
- simplex - Flag for simplex, otherwise use a tensor-product cell

  Output Parameters:
. ref     - the reference tree DMPlex object

  Level: beginner

.keywords: reference cell
.seealso: DMPlexSetReferenceTree(), DMPlexGetReferenceTree()
@*/
PetscErrorCode DMPlexCreateDefaultReferenceTree(MPI_Comm comm, PetscInt dim, PetscBool simplex, DM *ref)
{
  DM             K, Kref;
  PetscInt       p, pStart, pEnd, pRefStart, pRefEnd, d, offset;
  PetscInt      *permvals, *unionCones, *coneSizes, *unionOrientations, numUnionPoints, *numDimPoints, numCones, numVerts;
  DMLabel        identity, identityRef;
  PetscSection   unionSection, unionConeSection;
  PetscScalar   *unionCoords;
  IS             perm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* create a reference element */
  ierr = DMPlexCreateReferenceCell(comm, dim, simplex, &K);CHKERRQ(ierr);
  ierr = DMPlexCreateLabel(K, "identity");CHKERRQ(ierr);
  ierr = DMPlexGetLabel(K, "identity", &identity);CHKERRQ(ierr);
  ierr = DMPlexGetChart(K, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    ierr = DMLabelSetValue(identity, p, p);CHKERRQ(ierr);
  }
  /* refine it */
  ierr = DMRefine(K,comm,&Kref);CHKERRQ(ierr);

  /* the reference tree is the union of these two, without duplicating
   * points that appear in both */
  ierr = DMPlexGetLabel(Kref, "identity", &identityRef);CHKERRQ(ierr);
  ierr = DMPlexGetChart(Kref, &pRefStart, &pRefEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &unionSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(unionSection, 0, (pEnd - pStart) + (pRefEnd - pRefStart));CHKERRQ(ierr);
  /* count points that will go in the union */
  for (p = pStart; p < pEnd; p++) {
    ierr = PetscSectionSetDof(unionSection, p - pStart, 1);CHKERRQ(ierr);
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt q, qSize;
    ierr = DMLabelGetValue(identityRef, p, &q);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(identityRef, q, &qSize);CHKERRQ(ierr);
    if (qSize > 1) {
      ierr = PetscSectionSetDof(unionSection, p - pRefStart + (pEnd - pStart), 1);CHKERRQ(ierr);
    }
  }
  ierr = PetscMalloc1((pEnd - pStart) + (pRefEnd - pRefStart),&permvals);CHKERRQ(ierr);
  offset = 0;
  /* stratify points in the union by topological dimension */
  for (d = 0; d <= dim; d++) {
    PetscInt cStart, cEnd, c;

    ierr = DMPlexGetHeightStratum(K, d, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      permvals[offset++] = c;
    }

    ierr = DMPlexGetHeightStratum(Kref, d, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      permvals[offset++] = c + (pEnd - pStart);
    }
  }
  ierr = ISCreateGeneral(comm, (pEnd - pStart) + (pRefEnd - pRefStart), permvals, PETSC_OWN_POINTER, &perm);CHKERRQ(ierr);
  ierr = PetscSectionSetPermutation(unionSection,perm);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(unionSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(unionSection,&numUnionPoints);CHKERRQ(ierr);
  ierr = PetscMalloc2(numUnionPoints,&coneSizes,dim+1,&numDimPoints);CHKERRQ(ierr);
  /* count dimension points */
  for (d = 0; d <= dim; d++) {
    PetscInt cStart, cOff, cOff2;
    ierr = DMPlexGetHeightStratum(K,d,&cStart,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection,cStart-pStart,&cOff);CHKERRQ(ierr);
    if (d < dim) {
      ierr = DMPlexGetHeightStratum(K,d+1,&cStart,NULL);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(unionSection,cStart-pStart,&cOff2);CHKERRQ(ierr);
    }
    else {
      cOff2 = numUnionPoints;
    }
    numDimPoints[dim - d] = cOff2 - cOff;
  }
  ierr = PetscSectionCreate(comm, &unionConeSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(unionConeSection, 0, numUnionPoints);CHKERRQ(ierr);
  /* count the cones in the union */
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, uOff;

    ierr = DMPlexGetConeSize(K, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pStart,&uOff);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(unionConeSection, uOff, dof);CHKERRQ(ierr);
    coneSizes[uOff] = dof;
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt dof, uDof, uOff;

    ierr = DMPlexGetConeSize(Kref, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff);CHKERRQ(ierr);
    if (uDof) {
      ierr = PetscSectionSetDof(unionConeSection, uOff, dof);CHKERRQ(ierr);
      coneSizes[uOff] = dof;
    }
  }
  ierr = PetscSectionSetUp(unionConeSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(unionConeSection,&numCones);CHKERRQ(ierr);
  ierr = PetscMalloc2(numCones,&unionCones,numCones,&unionOrientations);CHKERRQ(ierr);
  /* write the cones in the union */
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, uOff, c, cOff;
    const PetscInt *cone, *orientation;

    ierr = DMPlexGetConeSize(K, p, &dof);CHKERRQ(ierr);
    ierr = DMPlexGetCone(K, p, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(K, p, &orientation);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pStart,&uOff);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionConeSection,uOff,&cOff);CHKERRQ(ierr);
    for (c = 0; c < dof; c++) {
      PetscInt e, eOff;
      e                           = cone[c];
      ierr                        = PetscSectionGetOffset(unionSection, e - pStart, &eOff);CHKERRQ(ierr);
      unionCones[cOff + c]        = eOff;
      unionOrientations[cOff + c] = orientation[c];
    }
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt dof, uDof, uOff, c, cOff;
    const PetscInt *cone, *orientation;

    ierr = DMPlexGetConeSize(Kref, p, &dof);CHKERRQ(ierr);
    ierr = DMPlexGetCone(Kref, p, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(Kref, p, &orientation);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff);CHKERRQ(ierr);
    if (uDof) {
      ierr = PetscSectionGetOffset(unionConeSection,uOff,&cOff);CHKERRQ(ierr);
      for (c = 0; c < dof; c++) {
        PetscInt e, eOff, eDof;

        e    = cone[c];
        ierr = PetscSectionGetDof(unionSection, e - pRefStart + (pEnd - pStart),&eDof);CHKERRQ(ierr);
        if (eDof) {
          ierr = PetscSectionGetOffset(unionSection, e - pRefStart + (pEnd - pStart), &eOff);CHKERRQ(ierr);
        }
        else {
          ierr = DMLabelGetValue(identityRef, e, &e);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(unionSection, e - pStart, &eOff);CHKERRQ(ierr);
        }
        unionCones[cOff + c]        = eOff;
        unionOrientations[cOff + c] = orientation[c];
      }
    }
  }
  /* get the coordinates */
  {
    PetscInt vStart, vEnd, vRefStart, vRefEnd, v, vDof, vOff;
    PetscSection KcoordsSec, KrefCoordsSec;
    Vec      KcoordsVec, KrefCoordsVec;
    PetscScalar *Kcoords;

    DMGetCoordinateSection(K, &KcoordsSec);CHKERRQ(ierr);
    DMGetCoordinatesLocal(K, &KcoordsVec);CHKERRQ(ierr);
    DMGetCoordinateSection(Kref, &KrefCoordsSec);CHKERRQ(ierr);
    DMGetCoordinatesLocal(Kref, &KrefCoordsVec);CHKERRQ(ierr);

    numVerts = numDimPoints[0];
    ierr     = PetscMalloc1(numVerts * dim,&unionCoords);CHKERRQ(ierr);
    ierr     = DMPlexGetDepthStratum(K,0,&vStart,&vEnd);CHKERRQ(ierr);

    offset = 0;
    for (v = vStart; v < vEnd; v++) {
      ierr = PetscSectionGetOffset(unionSection,v - pStart,&vOff);CHKERRQ(ierr);
      ierr = VecGetValuesSection(KcoordsVec, KcoordsSec, v, &Kcoords);CHKERRQ(ierr);
      for (d = 0; d < dim; d++) {
        unionCoords[offset * dim + d] = Kcoords[d];
      }
      offset++;
    }
    ierr = DMPlexGetDepthStratum(Kref,0,&vRefStart,&vRefEnd);CHKERRQ(ierr);
    for (v = vRefStart; v < vRefEnd; v++) {
      ierr = PetscSectionGetDof(unionSection,v - pRefStart + (pEnd - pStart),&vDof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(unionSection,v - pRefStart + (pEnd - pStart),&vOff);CHKERRQ(ierr);
      ierr = VecGetValuesSection(KrefCoordsVec, KrefCoordsSec, v, &Kcoords);CHKERRQ(ierr);
      if (vDof) {
        for (d = 0; d < dim; d++) {
          unionCoords[offset * dim + d] = Kcoords[d];
        }
        offset++;
      }
    }
  }
  ierr = DMCreate(comm,ref);CHKERRQ(ierr);
  ierr = DMSetType(*ref,DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*ref,dim);CHKERRQ(ierr);
  ierr = DMPlexCreateFromDAG(*ref,dim,numDimPoints,coneSizes,unionCones,unionOrientations,unionCoords);CHKERRQ(ierr);
  /* clean up */
  ierr = PetscSectionDestroy(&unionSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&unionConeSection);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = PetscFree(unionCoords);CHKERRQ(ierr);
  ierr = PetscFree2(unionCones,unionOrientations);CHKERRQ(ierr);
  ierr = PetscFree2(coneSizes,numDimPoints);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = DMDestroy(&Kref);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

