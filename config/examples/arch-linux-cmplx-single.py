#!/usr/bin/env python

configure_options = [
  '--with-scalar-type=complex',
  '--with-precision=single',
  '--download-make=1',
   ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
