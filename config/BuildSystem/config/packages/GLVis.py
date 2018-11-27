import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit              = 'fa3aeaa285589bd788f9d21679d0537381184f75'
    self.download               = ['git://https://github.com/stefanozampini/glvis.git']
    self.linkedbypetsc          = 0
    self.downloadonWindows      = 1
    self.cxx                    = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.x11    = framework.require('config.packages.X',self)
    self.opengl = framework.require('config.packages.opengl',self)
    self.deps   = [self.x11,self.opengl]
    return

  def updateGitDir(self):
    import os
    config.package.GNUPackage.updateGitDir(self)
    if not hasattr(self.sourceControl, 'git') or (self.packageDir != os.path.join(self.externalPackagesDir,'git.'+self.package)):
      return
    Dir = self.getDir()
    try:
      mfem = self.mfem
    except AttributeError:
      try:
        self.executeShellCommand([self.sourceControl.git, 'submodule', 'update', '--init'], cwd=Dir, log=self.log)
        import os
        if os.path.isfile(os.path.join(Dir,'mfem','README')):
          self.mfem = os.path.join(Dir,'mfem')
        else:
          raise RuntimeError
      except RuntimeError:
        raise RuntimeError('Could not initialize mfem submodule needed by GLVis')
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'glvis_config.mk'),'w')
    g.write('PREFIX = .\n')
    g.write('INSTALL = /usr/bin/install\n')
    g.write('AR = '+self.setCompilers.AR+'\n')
    g.write('MFEM_DIR = ./mfem\n')
    g.write('GLVIS_OPTS = \n')
    g.write('GLVIS_LDFLAGS = \n')
    g.write('GL_OPTS = '+self.headers.toString(self.x11.include)+'\n')
    g.write('GL_LIBS = '+self.libraries.toString(self.x11.lib)+' '+self.libraries.toString(self.opengl.lib)+'\n')
    g.write('GLVIS_USE_FREETYPE = NO\n')
    g.write('GLVIS_USE_LIBTIFF = NO\n')
    g.write('GLVIS_USE_LIBPNG = NO\n')

    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS = ' + self.removeWarningFlags(self.setCompilers.getCompilerFlags())+'\n')
    self.setCompilers.popLanguage()

    # build flags for serial MFEM
    self.setCompilers.pushLanguage('Cxx')
    mfem_flags='CXX=\"'+self.setCompilers.getCompiler()+'\" CXXFLAGS=\"-O3 '+self.setCompilers.getCompilerFlags()+'\"'
    self.setCompilers.popLanguage()

    g.write('PETSC_MFEM_FLAGS = '+mfem_flags+'\n')
    g.close()

    if self.installNeeded('glvis_config.mk'):
      try:
        self.logPrintBox('Compiling GLVis; this may take several minutes')
        output0,err0,ret0 = config.package.Package.executeShellCommand('cd '+self.packageDir+'/mfem && make clean && '+self.make.make_jnp+' serial '+mfem_flags, cwd=self.packageDir, timeout=2500, log = self.log)
        output1,err1,ret1 = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make clean && '+self.make.make_jnp+' GLVIS_CONFIG_MK=glvis_config.mk', cwd=self.packageDir, timeout=2500, log = self.log)
        installBinDir = os.path.join(self.installDir,'bin')
        self.logPrintBox('Installing GLVis; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output2,err2,ret2 = config.package.Package.executeShellCommandSeq(
          [[self.installSudo+'mkdir', '-p', installBinDir],
           self.installSudo+'cp -f glvis '+installBinDir+'/.',
           self.installSudo+'chmod 750 '+installBinDir+'/glvis'
          ], cwd=self.packageDir, timeout=50, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on GLVis: '+str(e))
      self.postInstall(output0+err0+output1+err1+output2+err2,'glvis_config.mk')

    return self.installDir
