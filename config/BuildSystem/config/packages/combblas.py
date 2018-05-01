import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'master'
    self.download          = ['git://https://bitbucket.org/berkeleylab/combinatorial-blas-2.0.git','https://bitbucket.org/berkeleylab/combinatorial-blas-2.0/get/'+self.gitcommit+'.tar.gz']
  #  self.functions         = []
#    self.functionsCxx      = 1
    self.includes          = ['CombBLAS/CombBLAS.h']
    self.liblist           = [['libCombBLAS.a','libGraphGenlib.a','libUsortlib.a']]
    self.hastests          = 1
    self.cxx               = 1
    self.requirescxx11     = 1
    self.cmakelistdir      = '/CombBLAS'
    self.downloaddirnames  = ['CombBLAS']


  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.deps          = [self.mpi, self.mathlib]


