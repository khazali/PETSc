import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions    = ['TBB_runtime_interface_version']
    self.includes     = ['tbb/blocked_range.h','tbb/parallel_for.h','tbb/task_scheduler_init.h']
    self.liblist      = [['libtbb.so']]
    self.cxx          = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries',self)
    self.deps = []
    return
