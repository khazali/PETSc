#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc-private/matorderimpl.h> /*I      "petscmat.h"      I*/

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateOrderingClosure_Static"
PetscErrorCode DMPlexCreateOrderingClosure_Static(DM dm, PetscInt numPoints, const PetscInt pperm[], PetscInt **clperm, PetscInt **invclperm)
{
  PetscInt      *perm, *iperm;
  PetscInt       depth, d, pStart, pEnd, fStart, fMax, fEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc2(pEnd-pStart,PetscInt,&perm,pEnd-pStart,PetscInt,&iperm);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) iperm[p] = -1;
  for (d = depth; d > 0; --d) {
    ierr = DMPlexGetDepthStratum(dm, d,   &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, d-1, &fStart, &fEnd);CHKERRQ(ierr);
    fMax = fStart;
    for (p = pStart; p < pEnd; ++p) {
      const PetscInt *cone;
      PetscInt        point, coneSize, c;

      if (d == depth) {
        perm[p]         = pperm[p];
        iperm[pperm[p]] = p;
      }
      point = perm[p];
      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        const PetscInt oldc = cone[c];
        const PetscInt newc = iperm[oldc];

        if (newc < 0) {
          perm[fMax]  = oldc;
          iperm[oldc] = fMax++;
        }
      }
    }
    if (fMax != fEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of depth %d faces %d does not match permuted nubmer %d", d, fEnd-fStart, fMax-fStart);
  }
  *clperm    = perm;
  *invclperm = iperm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetOrdering"
/*@
  DMPlexGetOrdering - Calculate a reordering of the mesh

  Collective on DM

  Input Parameter:
+ dm - The DMPlex object
- otype - type of reordering, one of the following:
$     MATORDERINGNATURAL - Natural
$     MATORDERINGND - Nested Dissection
$     MATORDERING1WD - One-way Dissection
$     MATORDERINGRCM - Reverse Cuthill-McKee
$     MATORDERINGQMD - Quotient Minimum Degree


  Output Parameter:
. perm - The point permutation as an IS

  Level: intermediate

.keywords: mesh
.seealso: MatGetOrdering()
@*/
PetscErrorCode DMPlexGetOrdering(DM dm, MatOrderingType otype, IS *perm)
{
  PetscInt       numCells = 0;
  PetscInt      *start = NULL, *adjacency = NULL, *cperm, *clperm, *invclperm, *mask, *xls, pStart, pEnd, c, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(perm, 3);
  ierr = DMPlexCreateNeighborCSR(dm, 0, &numCells, &start, &adjacency);CHKERRQ(ierr);
  ierr = PetscMalloc3(numCells,PetscInt,&cperm,numCells,PetscInt,&mask,numCells*2,PetscInt,&xls);CHKERRQ(ierr);
  /* Shift for Fortran numbering */
  for (i = 0; i < start[numCells]; ++i) ++adjacency[i];
  for (i = 0; i <= numCells; ++i)       ++start[i];
  ierr = SPARSEPACKgenrcm(&numCells, start, adjacency, cperm, mask, xls);CHKERRQ(ierr);
  ierr = PetscFree(start);CHKERRQ(ierr);
  ierr = PetscFree(adjacency);CHKERRQ(ierr);
  /* Shift for Fortran numbering */
  for (c = 0; c < numCells; ++c) --cperm[c];
  /* Construct closure */
  ierr = DMPlexCreateOrderingClosure_Static(dm, numCells, cperm, &clperm, &invclperm);
  ierr = PetscFree3(cperm,mask,xls);CHKERRQ(ierr);
  ierr = PetscFree(clperm);CHKERRQ(ierr);
  /* Invert permutation */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), pEnd-pStart, invclperm, PETSC_OWN_POINTER, perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPermute"
/*@
  DMPlexPermute - Reorder the mesh according to the input permutation

  Collective on DM

  Input Parameter:
+ dm - The DMPlex object
- perm - The point permutation

  Output Parameter:
. pdm - The permuted DM

  Level: intermediate

.keywords: mesh
.seealso: MatPermute()
@*/
PetscErrorCode DMPlexPermute(DM dm, IS perm, DM *pdm)
{
  DM_Plex       *plex = (DM_Plex *) dm->data, *plexNew;
  PetscSection   section, sectionNew;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(perm, IS_CLASSID, 2);
  PetscValidPointer(pdm, 3);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), pdm);CHKERRQ(ierr);
  ierr = DMSetType(*pdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*pdm, dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionPermute(section, perm, &sectionNew);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(*pdm, sectionNew);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionNew);CHKERRQ(ierr);
  plexNew = (DM_Plex *) (*pdm)->data;
  /* Ignore ltogmap, ltogmapb */
  /* Ignore sf, defaultSF */
  /* Ignore globalVertexNumbers, globalCellNumbers */
  /* Remap coordinates */
  {
    DM           cdm, cdmNew;
    PetscSection csection, csectionNew;
    Vec          coordinates, coordinatesNew;
    PetscScalar *coords, *coordsNew;
    PetscInt     pStart, pEnd, p;

    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &csection);CHKERRQ(ierr);
    ierr = PetscSectionPermute(csection, perm, &csectionNew);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecDuplicate(coordinates, &coordinatesNew);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecGetArray(coordinatesNew, &coordsNew);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(csectionNew, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, offNew, d;

      ierr = PetscSectionGetDof(csectionNew, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(csection, p, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(csectionNew, p, &offNew);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) coordsNew[offNew+d] = coords[off+d];
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecRestoreArray(coordinatesNew, &coordsNew);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(*pdm, &cdmNew);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(cdmNew, csectionNew);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*pdm, coordinatesNew);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&csectionNew);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinatesNew);CHKERRQ(ierr);
  }
  /* Reorder labels */
  {
    DMLabel label = plex->labels, labelNew = NULL;

    while (label) {
      if (!plexNew->labels) {
        ierr = DMLabelPermute(label, perm, &plexNew->labels);CHKERRQ(ierr);
        labelNew = plexNew->labels;
      } else {
        ierr = DMLabelPermute(label, perm, &labelNew->next);CHKERRQ(ierr);
      }
      label = label->next;
    }
    if (plex->subpointMap) {ierr = DMLabelPermute(plex->subpointMap, perm, &plexNew->subpointMap);CHKERRQ(ierr);}
  }
  /* Reorder topology */
  {
    const PetscInt *pperm;
    PetscInt        maxConeSize, maxSupportSize, n, pStart, pEnd, p;

    ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    plexNew->maxConeSize    = maxConeSize;
    plexNew->maxSupportSize = maxSupportSize;
    ierr = PetscSectionDestroy(&plexNew->coneSection);CHKERRQ(ierr);
    ierr = PetscSectionPermute(plex->coneSection, perm, &plexNew->coneSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(plexNew->coneSection, &n);CHKERRQ(ierr);
    ierr = PetscMalloc(n * sizeof(PetscInt), &plexNew->cones);CHKERRQ(ierr);
    ierr = PetscMalloc(n * sizeof(PetscInt), &plexNew->coneOrientations);CHKERRQ(ierr);
    ierr = ISGetIndices(perm, &pperm);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(plex->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, offNew, d;

      ierr = PetscSectionGetDof(plexNew->coneSection, pperm[p], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(plex->coneSection, p, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(plexNew->coneSection, pperm[p], &offNew);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        plexNew->cones[offNew+d]            = pperm[plex->cones[off+d]];
        plexNew->coneOrientations[offNew+d] = plex->coneOrientations[off+d];
      }
    }
    ierr = PetscSectionDestroy(&plexNew->supportSection);CHKERRQ(ierr);
    ierr = PetscSectionPermute(plex->supportSection, perm, &plexNew->supportSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(plexNew->supportSection, &n);CHKERRQ(ierr);
    ierr = PetscMalloc(n * sizeof(PetscInt), &plexNew->supports);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(plex->supportSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, offNew, d;

      ierr = PetscSectionGetDof(plexNew->supportSection, pperm[p], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(plex->supportSection, p, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(plexNew->supportSection, pperm[p], &offNew);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        plexNew->supports[offNew+d] = pperm[plex->supports[off+d]];
      }
    }
    ierr = ISRestoreIndices(perm, &pperm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
