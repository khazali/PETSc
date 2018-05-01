#requirespackage 'PETSC_HAVE_COMBBLAS'

ALL: lib

CFLAGS    = ${COMBBLAS_INCLUDE}
FFLAGS    =
SOURCECXX = combblas.cxx
SOURCEF	  =
SOURCEH	  =
LIBBASE	  = libpetscmat
DIRS	  =
MANSEC	  = Mat
LOCDIR	  = src/mat/impls/combblas/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test


