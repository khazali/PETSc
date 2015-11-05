import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.download     = ['http://www.hdfgroup.org/ftp/HDF5/prev-releases/hdf5-1.8.12/src/hdf5-1.8.12.tar.gz',
                         'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hdf5-1.8.12.tar.gz']
    self.functions = ['H5T_init']
    self.includes  = ['hdf5.h']
    self.liblist   = [['libhdf5_hl.a', 'libhdf5.a']]
    self.needsMath = 1
    self.needsCompression = 0
    self.complex          = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.mpi  = framework.require('config.packages.MPI',self)
    self.deps = [self.mpi]
    return

  def generateLibList(self, framework):
    '''First try library list without compression libraries (zlib) then try with'''
    list = []
    for l in self.liblist:
      list.append(l)
    if self.libraries.compression:
      for l in self.liblist:
        list.append(l + self.libraries.compression)
    self.liblist = list
    return config.package.Package.generateLibList(self,framework)

  def formCMakeConfigureArgs(self):
    ''' Add HDF5 specific HDF5_ENABLE_PARALLEL flag and enable Fortran if available '''
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DHDF5_ENABLE_PARALLEL=ON')
    args.append('-DHDF5_BUILD_CPP_LIB=OFF')
    args.append('-DHDF5_ENABLE_Z_LIB_SUPPORT=ON')
    if hasattr(self.compilers, 'FC'):
      args.append('-DHDF5_BUILD_FORTRAN=ON')
      args.append('-DHDF5_BUILD_HL_LIB=ON')
    if self.sharedLibraries.useShared:
      args.append('-DBUILD_SHARED_LIBS=ON')
    else:
      args.append('-DBUILD_SHARED_LIBS=OFF')
    return args

  def configureLibrary(self):
    if hasattr(self.compilers, 'FC'):
      # PETSc does not need the Fortran interface, but some users will call the Fortran interface
      # and expect our standard linking to be sufficient.  Thus we try to link the Fortran
      # libraries, but fall back to linking only C.
      self.liblist = [['libhdf5hl_fortran.a','libhdf5_fortran.a'] + libs for libs in self.liblist] + self.liblist
    config.package.CMakePackage.configureLibrary(self)
    if self.libraries.check(self.dlib, 'H5Pset_fapl_mpio'):
      self.addDefine('HAVE_H5PSET_FAPL_MPIO', 1)
    return
