#   Language is completely determined by the file prefix, .c .cxx .F
#     .F90 so this need not be in the requirements list
#
#   MPIUNI should work for all -n 1 examples so this need not be in the
#   requirements list
# 
#   DATAFILES are listed in the example arguments (e.g. -f
#   ${DATAFILES}/) so this is need not be in the requirement list
#
#   For packages, scalartypes and precisions:
#     ! => not
#     , => and
#
#   Precision types: single, double, quad, int32
#   Scalar types: complex  (and !complex)
#
#   Some examples:
#      requires:   x, superlu_dist, !single  
#      requires: !complex !single
#      requires: int32
#
#   There is some limited support for mapping args onto packages

makefileMap={}
makefileMap["TESTEXAMPLES_F90_NOCOMPLEX"]="requires: !complex"  # Need to check
makefileMap["TESTEXAMPLES_F2003"]="requires: define(USING_F2003)"
makefileMap["TESTEXAMPLES_C_COMPLEX"]="requires: complex"
makefileMap["TESTEXAMPLES_FORTRAN_COMPLEX"]="requires: complex"
makefileMap["TESTEXAMPLES_C_NOCOMPLEX"]="requires: !complex"
makefileMap["TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE"]="requires: !complex, !single"
makefileMap["TESTEXAMPLES_FORTRAN_NOCOMPLEX"]="requires: !complex"
makefileMap["TESTEXAMPLES_C_X"]="requires: x"
makefileMap["TESTEXAMPLES_C_X_MPIUNI"]="requires: x"
makefileMap["TESTEXAMPLES_C_NOTSINGLE"]="requires: !single"
makefileMap['TESTEXAMPLES_INFO']="requires: define(USE_INFO)"
makefileMap['TESTEXAMPLES_NOTSINGLE']="requires: !single"
makefileMap["TESTEXAMPLES_CUDA"]="requires: cuda"
makefileMap["TESTEXAMPLES_CUSP"]="requires: cusp"
makefileMap["TESTEXAMPLES_CUSPARSE"]="requires: cusparse"
makefileMap["TESTEXAMPLES_HYPRE"]="requires: hypre"
makefileMap["TESTEXAMPLES_MUMPS"]="requires: mumps"
makefileMap["TESTEXAMPLES_SUPERLU"]="requires: superlu"
makefileMap["TESTEXAMPLES_SUPERLU_DIST"]="requires: superlu_dist"
makefileMap["TESTEXAMPLES_MKL_PARDISO"]="requires: mkl_pardiso"
makefileMap["TESTEXAMPLES_MOAB"]="requires: moab"
makefileMap["TESTEXAMPLES_THREADCOMM"]="requires: threadcomm"

argMap={}
#packages="superlu superlu_dist hypre strumpack elemental cuda cusp mkl_pardiso moab threadcomm"
# Figure out superlu later
packages="superlu_dist hypre strumpack elemental cuda cusp mkl_pardiso moab threadcomm"
for pkg in packages.split():
  argMap[pkg]="requires: "+pkg
