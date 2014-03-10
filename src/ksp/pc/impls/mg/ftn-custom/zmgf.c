#include <petsc-private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcmgsetuplevels_           PCMGSETUPLEVELS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcmgsetuplevels_           pcmgsetuplevels
#endif

PETSC_EXTERN void PETSC_STDCALL pcmgsetuplevels_(PC *pc,MPI_Comm *comms, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(comms);
  *ierr = PCMGSetUpLevels(*pc,comms);
}

