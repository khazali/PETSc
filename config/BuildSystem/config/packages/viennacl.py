import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['http://downloads.sourceforge.net/project/viennacl/1.7.x/ViennaCL-1.7.0.tar.gz',
                            'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/ViennaCL-1.7.0.tar.gz' ]
    self.downloadfilename = str('ViennaCL-1.7.0')
    self.includes        = ['viennacl/forwards.h']
    self.forceLanguage   = 'CUDA'
    self.cxx             = 0
    self.downloadonWindows = 1
    self.complex          = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda  = framework.require('config.packages.cuda',self)
    self.deps = [self.cuda]
    return

  def Install(self):
    import shutil
    import os
    self.log.write('ViennaCLDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    #includeDir = self.packageDir
    srcdir     = os.path.join(self.packageDir, 'viennacl')
    destdir    = os.path.join(self.installDir, 'include', 'viennacl')
    if self.installSudo:
      self.installDirProvider.printSudoPasswordMessage()
      try:
        output,err,ret  = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+destdir+' && '+self.installSudo+'rm -rf '+destdir+'  && '+self.installSudo+'cp -rf '+srcdir+' '+destdir, timeout=6000, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error copying ViennaCL include files from '+os.path.join(self.packageDir, 'ViennaCL')+' to '+packageDir)
    else:
      try:
        if os.path.isdir(destdir): shutil.rmtree(destdir)
        shutil.copytree(srcdir,destdir)
      except RuntimeError,e:
        raise RuntimeError('Error installing ViennaCL include files: '+str(e))
    return self.installDir

  def checkCUSPVersion(self):
    if 'known-cusp-version' in self.argDB:
      if self.argDB['known-cusp-version'] < self.CUSPVersion:
        raise RuntimeError('CUSP version error '+self.argDB['known-cusp-version']+' < '+self.CUSPVersion+': PETSC currently requires CUSP version '+self.CUSPVersionStr+' or higher')
    elif not self.argDB['with-batch']:
      self.pushLanguage('CUDA')
      oldFlags = self.compilers.CUDAPPFLAGS
      self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
      if not self.checkRun('#include <cusp/version.h>\n#include <stdio.h>', 'if (CUSP_VERSION < ' + self.CUSPVersion +') {printf("Invalid version %d\\n", CUSP_VERSION); return 1;}'):
        raise RuntimeError('CUSP version error: PETSC currently requires CUSP version '+self.CUSPVersionStr+' or higher.')
      self.compilers.CUDAPPFLAGS = oldFlags
      self.popLanguage()
    else:
      raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-cusp-version')
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does a additional tests needed by ViennaCL'''
    config.package.Package.configureLibrary(self)
    self.pushLanguage('CUDA')
    oldFlags = self.compilers.CUDAPPFLAGS
    self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
    if not self.checkRun('#include <viennacl/version.hpp>\n#include <stdio.h>', 'if (VIENNACL_MINOR_VERSION < 6) {printf("Invalid version %d\\n", VIENNACL_MINOR_VERSION); return 1;}'):
      raise RuntimeError('ViennaCL version error: PETSC currently requires ViennaCL version 1.6.0 or higher.')
    self.compilers.CUDAPPFLAGS = oldFlags
    self.popLanguage()
    return
