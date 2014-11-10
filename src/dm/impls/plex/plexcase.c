#define PETSCDM_DLL
#include <petsc-private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

/* Utility struct to store the contents of a Case file in memory */
typedef struct {
  int   index;    /* Type of section */
  int   zoneID;
  int   first;
  int   last;
  int   type;
  int   nd;       /* Either ND or element-type */
  void *data;
} CaseSection;

#undef __FUNCT__
#define __FUNCT__ "DMPlexCaseRead"
PetscErrorCode DMPlexCaseRead(PetscViewer viewer, char *buffer, char delim)
{
  PetscInt i = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  do {ierr = PetscViewerRead(viewer, &(buffer[i++]), 1, PETSC_CHAR);CHKERRQ(ierr);}
  while (buffer[i-1] != '\0' && buffer[i-1] != delim);
  buffer[i] = '\0';
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateCase_ReadSection"
PetscErrorCode DMPlexCreateCase_ReadSection(PetscViewer viewer, CaseSection *s)
{
  char           buffer[PETSC_MAX_PATH_LEN];
  int            snum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Fast-forward to next section and derive its index */
  ierr = DMPlexCaseRead(viewer, buffer, '(');CHKERRQ(ierr);
  ierr = DMPlexCaseRead(viewer, buffer, ' ');CHKERRQ(ierr);
  snum = sscanf(buffer, "%d", &(s->index));
  /* If we can't match an index return -1 to signal end-of-file */
  if (snum < 1) {s->index = -1;   PetscFunctionReturn(0);}

  switch (s->index) {
  case 0:    /* Comment */
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    break;
  case 2:    /* Dimension */
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "%d", &(s->nd));CHKERRQ(snum!=1);
    break;
  case 10:   /* Vertices */
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "(%x %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));CHKERRQ(snum!=5);
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    break;
  case 12:   /* Cells */
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "(%x", &(s->zoneID));CHKERRQ(snum!=1);
    if (s->zoneID == 0) {snum = sscanf(buffer, "(%x %x %x %d)", &(s->zoneID), &(s->first), &(s->last), &(s->nd));CHKERRQ(snum!=4);}
    else {snum = sscanf(buffer, "(%x %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));CHKERRQ(snum!=5);}
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    break;
  case 13:   /* Faces */
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "(%x", &(s->zoneID));CHKERRQ(snum!=1);
    if (s->zoneID == 0) {
      snum = sscanf(buffer, "(%x %x %x %d)", &(s->zoneID), &(s->first), &(s->last), &(s->nd));CHKERRQ(snum!=4);
      ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    } else {
      PetscInt f, e, numEntries, numFaces, *face_array;
      int      entry;
      snum = sscanf(buffer, "(%x %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));CHKERRQ(snum!=5);
      switch (s->nd) {
      case 0: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mixed faces in Case files are not supported");
      case 2: numEntries = 2 + 2; break;  /* linear */
      case 3: numEntries = 2 + 3; break;  /* triangular */
      case 4: numEntries = 2 + 4; break;  /* quadrilateral */
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown face type in Case file");
      }
      ierr = DMPlexCaseRead(viewer, buffer, '(');CHKERRQ(ierr);
      ierr = DMPlexCaseRead(viewer, buffer, '\n');CHKERRQ(ierr);
      numFaces = s->last-s->first + 1;
      ierr = PetscMalloc1(numEntries*numFaces, &face_array);CHKERRQ(ierr);
      for (f = 0; f < numFaces; f++) {
        for (e = 0; e < numEntries; e++) {
          ierr = PetscViewerRead(viewer, buffer, 1, PETSC_STRING);CHKERRQ(ierr);
          snum = sscanf(buffer, "%x", &entry);CHKERRQ(snum!=1);
          face_array[f*numEntries + e] = entry;
        }
      }
      s->data = face_array;
    }
  default:
    ierr = DMPlexCaseRead(viewer, buffer, '\n');CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateCase"
/*@C
  DMPlexCreateCase - Create a DMPlex mesh from a FLUENT Case file.

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
. viewer - The Viewer associated with a Gmsh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: https://www.sharcnet.ca/Software/TGrid/pdf/ug/appb.pdf

  Level: beginner

.keywords: mesh, fluent, case
.seealso: DMPLEX, DMCreate()
@*/
PetscErrorCode DMPlexCreateCase(MPI_Comm comm, PetscViewer viewer, PetscBool interpolate, DM *dm)
{
  PetscMPIInt    rank;
  PetscInt       c, f, v, dim = -1, numCells = -1, numVertices = -1, numFaces = -1;
  PetscInt       numCellVertices = -1 , numFaceEntries = -1, numFaceVertices = -1;
  PetscInt      *faces, *cellVertices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  if (!rank) {
    PetscInt index;
    do {
      CaseSection s;
      ierr = DMPlexCreateCase_ReadSection(viewer, &s);CHKERRQ(ierr);
      index = s.index;
      switch (index) {
      case -1: break;  /* End-of-file */
      case 0: break;   /* Comment */
      case 2:          /* Dimsion */
        dim = s.nd;
        break;
      case 10:         /* Vertices */
        if (s.zoneID == 0) numVertices = s.last;
        break;
      case 12:         /* Cells */
        if (s.zoneID == 0) numCells = s.last;
        else {
          switch (s.nd) {
          case 0: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mixed elements in Case files are not supported");
          case 1: numCellVertices = 3; break;  /* triangular */
          case 2: numCellVertices = 4; break;  /* tetrahedral */
          case 3: numCellVertices = 4; break;  /* quadrilateral */
          case 4: numCellVertices = 8; break;  /* hexahedral */
          case 5: numCellVertices = 5; break;  /* pyramid */
          case 6: numCellVertices = 6; break;  /* wedge */
          default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell element-type in Case file");
          }
        }
        break;
      case 13:         /* Faces */
        if (s.zoneID == 0) numFaces = s.last - s.first + 1;
        else {
          if (s.nd == 0 || (numFaceEntries > 0 && s.nd != numFaceVertices)) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mixed facets in Case files are not supported");
          }
          if (numFaces < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No header stion for facets in Case file");
          if (numFaceVertices < 0) {
            numFaceVertices = s.nd;
            numFaceEntries = numFaceVertices + 2;
            ierr = PetscMalloc1(numFaces*numFaceEntries, &faces);CHKERRQ(ierr);
          }
          ierr = PetscMemcpy(&(faces[(s.first-1)*numFaceEntries]), s.data, (s.last-s.first+1)*numFaceEntries*sizeof(PetscInt));CHKERRQ(ierr);
          ierr = PetscFree(s.data);CHKERRQ(ierr);
        }
        break;
      }
    } while (index >= 0);
  }
  if (dim < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Case file does not include dimension");

  /* Allocate cell-vertex mesh */
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(*dm, 0, numCells + numVertices);CHKERRQ(ierr);
  if (!rank) {
    for (c = 0; c < numCells; ++c) {ierr = DMPlexSetConeSize(*dm, c, numCellVertices);CHKERRQ(ierr);}
  }
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  if (!rank) {
    /* Derive cell-vertex list from face-vertex and face-cell maps */
    if (numCells < 0 || numCellVertices < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Insufficent cell header information in Case file");
    ierr = PetscMalloc1(numCells*numCellVertices, &cellVertices);CHKERRQ(ierr);
    for (c = 0; c < numCells*numCellVertices; c++) cellVertices[c] = -1;
    for (f = 0; f < numFaces; f++) {
      PetscInt *cell;
      const PetscInt cl = faces[f*numFaceEntries + numFaceVertices];
      const PetscInt cr = faces[f*numFaceEntries + numFaceVertices + 1];
      const PetscInt *face = &(faces[f*numFaceEntries]);

      if (cl > 0) {
        cell = &(cellVertices[(cl-1) * numCellVertices]);
        for (v = 0; v < numFaceVertices; v++) {
          PetscBool found = PETSC_FALSE;
          for (c = 0; c < numCellVertices; c++) {
            if (cell[c] < 0) break;
            if (cell[c] == face[v]-1 + numCells) {found = PETSC_TRUE; break;}
          }
          if (!found) cell[c] = face[v]-1 + numCells;
        }
      }
      if (cr > 0) {
        cell = &(cellVertices[(cr-1) * numCellVertices]);
        for (v = 0; v < numFaceVertices; v++) {
          PetscBool found = PETSC_FALSE;
          for (c = 0; c < numCellVertices; c++) {
            if (cell[c] < 0) break;
            if (cell[c] == face[v]-1 + numCells) {found = PETSC_TRUE; break;}
          }
          if (!found) cell[c] = face[v]-1 + numCells;
        }
      }
    }
    for (c = 0; c < numCells; c++) {
      ierr = DMPlexSetCone(*dm, c, &(cellVertices[c*numCellVertices]));CHKERRQ(ierr);
    }
  }
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  if (interpolate) {
    DM idm = NULL;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }

  ierr = PetscFree2(cellVertices, faces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
