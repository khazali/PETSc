!
!
!  Include file for Fortran use of the IS (index set) package in PETSc
!
#include "petsc/finclude/petscis.h"

      type tIS
        PetscFortranAddr:: v
      end type tIS
      type tISColoring
        PetscFortranAddr:: v
      end type tISColoring
      type tPetscSection
        PetscFortranAddr:: v
      end type tPetscSection
      type tPetscSectionSym
        PetscFortranAddr:: v
      end type tPetscSectionSym

      IS, parameter :: PETSC_NULL_IS = tIS(-1)

      PetscEnum IS_COLORING_GLOBAL
      PetscEnum IS_COLORING_LOCAL
      parameter (IS_COLORING_GLOBAL = 0,IS_COLORING_LOCAL = 1)

      PetscEnum IS_GENERAL
      PetscEnum IS_STRIDE
      PetscEnum IS_BLOCK
      parameter (IS_GENERAL = 0,IS_STRIDE = 1,IS_BLOCK = 2)

      PetscEnum IS_GTOLM_MASK
      PetscEnum IS_GTOLM_DROP
      parameter (IS_GTOLM_MASK =0,IS_GTOLM_DROP = 1)

!
!  End of Fortran include file for the IS package in PETSc

