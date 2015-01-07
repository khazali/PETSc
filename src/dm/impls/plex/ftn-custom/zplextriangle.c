#include <petsc-private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexcreatetrianglefromfile_  DMPLEXCREATETRIANGLEFROMFILE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexcreatetrianglefromfile_  dmplexcreatetrianglefromfile
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexcreatetrianglefromfile_(MPI_Fint *comm, CHAR name PETSC_MIXED_LEN(lenN), PetscBool *interpolate, DM *dm, int *ierr PETSC_END_LEN(lenN))
{
  char *filename;

  FIXCHAR(name, lenN, filename);
  *ierr = DMPlexCreateTriangleFromFile(MPI_Comm_f2c(*(comm)), filename, *interpolate, dm);
  FREECHAR(name, filename);
}
