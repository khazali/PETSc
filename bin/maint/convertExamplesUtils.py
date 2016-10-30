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
#   buildrequires => file requires things just to build.  Usually
#      because of includes
#
#   There is some limited support for mapping args onto packages


makefileMap={}

# This looks for the pattern matching and then determines the
# requirement.  The distinction between buildrequires and requires is
# tricky.  I looked at the makefile's and files themselves to try and
# figure it out.
makefileMap["_COMPLEX"]="buildrequires: complex"
makefileMap["_NOCOMPLEX"]="buildrequires: !complex"
makefileMap["_NOTSINGLE"]="buildrequires: !single"
makefileMap["_NOSINGLE"]="buildrequires: !single"

makefileMap["_DOUBLEINT32"]="buildrequires: !define(USE_64BIT_INDICES) define(PETSC_USE_REAL_DOUBLE)"  
makefileMap["_THREADSAFETY"]="buildrequires: define(PETSC_USING_FREEFORM) define(PETSC_USING_F90)"
makefileMap["_F2003"]="buildrequires: define(PETSC_USING_FREEFORM) define(PETSC_USING_F2003)"
#makefileMap["_F90_DATATYPES"]="" # ??

makefileMap["_DATAFILESPATH"]="requires: datafilespath"
makefileMap['_INFO']="requires: define(USE_INFO)"

# Typo
makefileMap["_PARAMETIS"]="requires: parmetis"

# Some packages are runtime, but others are buildtime because of includes
reqpkgs=["CHOMBO", "CTETGEN", "ELEMENTAL","EXODUSII", "HDF5", "HYPRE", "LUSOL", "MATLAB", "MATLAB_ENGINE", "MKL_PARDISO", "ML", "MUMPS", "PARMETIS", "PARMS", "PASTIX", "PTSCOTCH", "REVOLVE", "SAWS", "SPAI", "STRUMPACK", "SUITESPARSE", "SUPERLU", "SUPERLU_DIST", "TRIANGLE", "TRILINOS", "YAML"]

bldpkgs=["MOAB", "FFTW", "TCHEM","VECCUDA","CUSP","CUSPARSE","X"]

for pkg in reqpkgs: makefileMap["_"+pkg]="requires: "+ pkg.lower()
for pkg in bldpkgs: makefileMap["_"+pkg]="buildrequires: "+ pkg.lower()

#  Map of "string" in arguments to package requirements; i.e.,
#    argMap[patternString]=packageRequired
#
argMap={}
for pkg in reqpkgs+bldpkgs:
  argMap[pkg]="requires: "+pkg.lower()
argMap['DATAFILESPATH']='requires: datafilespath'
