!
!  Include file for Fortran use of the PetscViewer package in PETSc
!
#include "petsc/finclude/petscviewer.h"

      type tPetscViewer
        PetscFortranAddr:: v
      end type tPetscViewer

      PetscViewer, parameter :: PETSC_NULL_VIEWER                          &
     &            = tPetscViewer(-1)
!
!     The numbers used below should match those in
!     petsc/private/fortranimpl.h
!
      PetscViewer, parameter :: PETSC_VIEWER_STDOUT_SELF =                &
     &           tPetscViewer(9)
      PetscViewer, parameter :: PETSC_VIEWER_DRAW_WORLD   =                &
     &           tPetscViewer(4)
      PetscViewer, parameter :: PETSC_VIEWER_DRAW_SELF    =                &
     &           tPetscViewer(5)
      PetscViewer, parameter :: PETSC_VIEWER_SOCKET_WORLD =                &
     &           tPetscViewer(6)
      PetscViewer, parameter :: PETSC_VIEWER_SOCKET_SELF  =                &
     &           tPetscViewer(7)
      PetscViewer, parameter :: PETSC_VIEWER_STDOUT_WORLD =                &
     &           tPetscViewer(8)
      PetscViewer, parameter :: PETSC_VIEWER_STDERR_WORLD =                &
     &           tPetscViewer(10)
      PetscViewer, parameter :: PETSC_VIEWER_STDERR_SELF  =                &
     &           tPetscViewer(11)
      PetscViewer, parameter :: PETSC_VIEWER_BINARY_WORLD =                &
     &           tPetscViewer(12)
      PetscViewer, parameter :: PETSC_VIEWER_BINARY_SELF  =                &
     &           tPetscViewer(13)
      PetscViewer, parameter :: PETSC_VIEWER_MATLAB_WORLD =                &
     &           tPetscViewer(14)
      PetscViewer, parameter :: PETSC_VIEWER_MATLAB_SELF  =                &
     &           tPetscViewer(15)

      PetscViewer PETSC_VIEWER_STDOUT_
      external PETSC_VIEWER_STDOUT_
      external PetscViewerAndFormatDestroy
!
!  Flags for binary I/O
!
      PetscEnum FILE_MODE_READ
      PetscEnum FILE_MODE_WRITE
      PetscEnum FILE_MODE_APPEND
      PetscEnum FILE_MODE_UPDATE
      PetscEnum FILE_MODE_APPEND_UPDATE

      parameter (FILE_MODE_READ = 0)
      parameter (FILE_MODE_WRITE = 1)
      parameter (FILE_MODE_APPEND = 2)
      parameter (FILE_MODE_UPDATE = 3)
      parameter (FILE_MODE_APPEND_UPDATE = 4)

!  End of Fortran include file for the PetscViewer package in PETSc







