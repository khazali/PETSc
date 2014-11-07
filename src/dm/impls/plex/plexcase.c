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
  PetscInt       i;
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
    snum = sscanf(buffer, "(%d %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));CHKERRQ(snum!=5);
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    break;
  case 12:   /* Cells */
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "(%d", &(s->zoneID));CHKERRQ(snum!=1);
    if (s->zoneID == 0) {snum = sscanf(buffer, "(%d %x %x %d)", &(s->zoneID), &(s->first), &(s->last), &(s->nd));CHKERRQ(snum!=4);}
    else {snum = sscanf(buffer, "(%d %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));CHKERRQ(snum!=5);}
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    break;
  case 13:   /* Faces */
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "(%d", &(s->zoneID));CHKERRQ(snum!=1);
    if (s->zoneID == 0) {snum = sscanf(buffer, "(%d %x %x %d)", &(s->zoneID), &(s->first), &(s->last), &(s->nd));CHKERRQ(snum!=4);}
    else {snum = sscanf(buffer, "(%d %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));CHKERRQ(snum!=5);}
    ierr = DMPlexCaseRead(viewer, buffer, ')');CHKERRQ(ierr);
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
  PetscInt       dim = -1, c, numCells, numVertices, numCellVertices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  if (!rank) {
    PetscInt index;
    do {
      CaseSection sec;
      ierr = DMPlexCreateCase_ReadSection(viewer, &sec);CHKERRQ(ierr);
      index = sec.index;
      switch (index) {
      case -1: break;  /* End-of-file */
      case 0: break;   /* Comment */
      case 2:          /* Dimension */
        dim = sec.nd;
        break;
      case 10      :   /* Vertices */
        if (sec.zoneID == 0) numVertices = sec.last;
        break;
      case 12:         /* Cells */
        if (sec.zoneID == 0) numCells = sec.last;
        else {
          switch (sec.nd) {
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
      case 13:
        break;
      case 45: break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown section index in Case file: %d", index);
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
  /* Add cell-vertex connections */

  PetscFunctionReturn(0);
}
