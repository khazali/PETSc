#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectgetcomm_        PETSCOBJECTGETCOMM
#define petscobjectsetcomm_        PETSCOBJECTSETCOMM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectgetcomm_        petscobjectgetcomm
#define petscobjectsetcomm_        petscobjectsetcomm
#endif

PETSC_EXTERN void PETSC_STDCALL petscobjectgetcomm_(PetscObject *obj,int *comm,PetscErrorCode *ierr)
{
  MPI_Comm c;
  *ierr = PetscObjectGetComm(*obj,&c);
  *(int*)comm =  MPI_Comm_c2f(c);
}

PETSC_EXTERN void PETSC_STDCALL petscobjectsetcomm_(PetscObject *obj,int *comm,PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetComm(*obj,MPI_Comm_f2c(*comm));
}

