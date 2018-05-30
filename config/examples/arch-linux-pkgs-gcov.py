#!/usr/bin/env python

configure_options = [
  '--with-gcov=1',
  #'--download-mpich=1', use system MPI as elemental fails with this
  '--download-fblaslapack=1',
  '--download-hypre=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-suitesparse=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-elemental=1',
  '--with-cxx-dialect=C++11',
  '--download-spai=1',
  '--download-parms=1',
  '--download-chaco=1',
  '--download-pastix',
  '--download-ctetgen',
  '--download-netcdf',
  '--download-hdf5',
  '--with-zlib=1',
  #'--download-exodusii', disabling due to pnetcdf+exodus errors that come up with this build
  #'--download-pnetcdf',
  '--download-party',
  '--download-yaml',
  '--download-ml',
  '--download-sundials',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
