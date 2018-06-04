PETSc for Python
================

Python bindings for PETSc.

Install
-------

If you have a working MPI implementation and the ``mpicc`` compiler
wrapper is on your search path, it highly recommended to install
``mpi4py`` first::

  $ pip install mpi4py

Ensure you have NumPy installed::

  $ pip install numpy

and finally::

  $ pip install petsc petsc4py

You can also install the in-development version of petsc4py with::

  $ pip install Cython numpy mpi4py
  $ pip install --no-deps git+https://bitbucket.org/petsc/petsc
  $ pip install --no-deps git+https://bitbucket.org/petsc/petsc4py

or::

  $ pip install Cython numpy mpi4py
  $ pip install --no-deps https://bitbucket.org/petsc/petsc/get/master.tar.gz
  $ pip install --no-deps https://bitbucket.org/petsc/petsc4py/get/master.tar.gz


Citations
---------

If PETSc for Python been significant to a project that leads to an
academic publication, please acknowledge that fact by citing the
project.

* L. Dalcin, P. Kler, R. Paz, and A. Cosimo,
  *Parallel Distributed Computing using Python*,
  Advances in Water Resources, 34(9):1124-1139, 2011.
  http://dx.doi.org/10.1016/j.advwatres.2011.04.013

* S. Balay, S. Abhyankar, M. Adams, J. Brown,
  P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W. Gropp,
  D. Karpeyev, D. Kaushik, M. Knepley, D. May, L. Curfman McInnes,
  R. Mills, T. Munson, K. Rupp, P. Sanan, B. Smith, S. Zampini,
  H. Zhang, and H. Zhang,
  *PETSc Users Manual*, ANL-95/11 - Revision 3.9, 2018.
  http://www.mcs.anl.gov/petsc/petsc-current/docs/manual.pdf
