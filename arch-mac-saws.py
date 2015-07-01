#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-openmpi',
    '--with-cc=gcc',
    '--with-cxx=g++',
    '--with-fc=0',
    '--download-cmake',
    '--download-saws',
    '--with-afterimage',
    'PETSC_ARCH=arch-mac-saws',
  ]
  configure.petsc_configure(configure_options)
