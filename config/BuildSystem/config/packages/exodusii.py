import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = '208a03eae9b1f384e199e78c765a5300aa2d4873'
    self.download          = ['git://https://github.com/gsjaardema/seacas.git','https://github.com/gsjaardema/seacas/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['seacas']
    self.functions         = ['ex_close']
    self.includes          = ['exodusII.h']
    self.liblist           = [['libexodus.a'], ]
    self.hastests          = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.netcdf = framework.require('config.packages.netcdf', self)
    self.hdf5   = framework.require('config.packages.hdf5', self)
    self.deps   = [self.netcdf, self.hdf5]
    return

  def configureLibrary(self):
    self.liblist = [['libexodus.a'], ['libexoIIv2c.a']]
    if hasattr(self.compilers, 'FC'):
      self.liblist.append(['libexoIIv2for.a'])
    config.package.Package.configureLibrary(self)

  def formCMakeConfigureArgs(self):
    import os
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build exodusII\nSuggest adding --download-cmake to ./configure arguments')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    args.append('-DACCESSDIR:PATH='+self.installDir)
    args.append('-DCMAKE_INSTALL_PREFIX:PATH='+self.installDir)
    args.append('-DCMAKE_INSTALL_RPATH:PATH='+os.path.join(self.installDir,'lib'))
    self.setCompilers.pushLanguage('C')
    args.append('-DCMAKE_C_COMPILER:FILEPATH="'+self.setCompilers.getCompiler()+'"')
    self.setCompilers.popLanguage()
    # building the fortran library is technically not required to add exodus support
    # we build it anyway so that fortran users can still use exodus functions directly 
    # from their code
    if hasattr(self.setCompilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      args.append('-DCMAKE_Fortran_COMPILER:FILEPATH="'+self.setCompilers.getCompiler()+'"')
      args.append('-DSEACASProj_ENABLE_SEACASExodus_for=ON')
      self.setCompilers.popLanguage()
    else:
      args.append('-DSEACASProj_ENABLE_SEACASExodus_for=OFF')
    args.append('-DSEACASProj_ENABLE_SEACASExodus=ON')
    args.append('-DSEACASProj_ENABLE_SEACASExoIIv2for32=OFF')
    args.append('-DSEACASProj_ENABLE_TESTS=ON')
    args.append('-DSEACASProj_SKIP_FORTRANCINTERFACE_VERIFY_TEST:BOOL=ON')
    args.append('-DTPL_ENABLE_Matio:BOOL=OFF')
    args.append('-DTPL_ENABLE_Netcdf:BOOL=ON')
    args.append('-DTPL_ENABLE_MPI=OFF')
    args.append('-DTPL_ENABLE_Pamgen=OFF')
    args.append('-DTPL_ENABLE_CGNS:BOOL=OFF')
    args.append('-DNetCDF_DIR:PATH='+self.netcdf.directory)
    args.append('-DHDF5_DIR:PATH='+self.hdf5.directory)
    if self.checkSharedLibrariesEnabled():
      args.append('-DBUILD_SHARED_LIBS:BOOL=ON')
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=Debug')
    else:
      args.append('-DCMAKE_BUILD_TYPE=Release')
    return args
