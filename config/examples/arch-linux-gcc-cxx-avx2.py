#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-debugging=0',
    '--with-clanguage=cxx',
    'COPTFLAGS=-mavx2 -g -O',
    'CXXOPTFLAGS=-mavx2 -g -O',
    'FOPTFLAGS=-mavx2 -g -O',
    '--with-mpi-dir=/homes/petsc/soft/gcc-avx2/mpich-3.3b1',
    '--with-blaslapack-dir=/homes/petsc/soft/gcc-avx2/fblaslapack-3.4.2',
    '--with-memalign=64',
    '--download-metis=1',
    '--download-parmetis=1',
    '--download-superlu_dist=1',
    '--with-cxx-dialect=C++11',
  ]
  configure.petsc_configure(configure_options)
