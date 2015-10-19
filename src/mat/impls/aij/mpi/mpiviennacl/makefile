#requirespackage 'PETSC_HAVE_VIENNACL'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = mpiaijviennacl.cu
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/mpi/mpiviennacl/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
