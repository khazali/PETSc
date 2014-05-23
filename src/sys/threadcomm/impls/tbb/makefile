#requirespackage   'PETSC_HAVE_TBB'

ALL: lib

CFLAGS   =
FFLAGS   =
CXXFLAGS = -I/soft/com/packages/intel/13/update5/tbb/include/tbb/
SOURCECXX = tctbb.cxx
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscsys
MANSEC   = PetscThreadComm
LOCDIR   = src/sys/threadcomm/impls/tbb/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
