import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://www.cs.sandia.gov/~kddevin/Zoltan_Distributions/zoltan_distrib_v3.8.tar.gz',
                      'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/zoltan_distrib_v3.8.tar.gz']
    self.functions = ['Zoltan_LB_Partition']
    self.includes  = ['zoltan.h']
    self.liblist   = [['libzoltan.a']]
    self.license   = 'http://www.cs.sandia.gov/Zoltan/Zoltan.html'
    self.needsMath = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)
    self.deps            = [self.mpi,]
    return

  def Install(self):
    import os
    self.framework.pushLanguage('C')
    args = ['--prefix='+self.installDir]
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    ccompiler=self.framework.getCompiler()
    args.append('CC="'+ccompiler+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    args.append('--enable-mpi')
    # use --with-mpi if we know it works
    if self.mpi.directory and (os.path.realpath(ccompiler)).find(os.path.realpath(self.mpi.directory)) >=0:
      self.framework.log.write('Zoltan configure: using --with-mpi-root='+self.mpi.directory+'\n')
      args.append('--with-mpi="'+self.mpi.directory+'"')
    # else provide everything!
    else:
      #print a message if the previous check failed
      if self.mpi.directory:
        self.framework.log.write('Zoltan configure: --with-mpi-dir specified - but could not use it\n')
        self.framework.log.write(str(os.path.realpath(ccompiler))+' '+str(os.path.realpath(self.mpi.directory))+'\n')

      args.append('--without-mpicc')
      if self.mpi.include:
        args.append('--with-mpi-incdir="'+self.mpi.include[0]+'"')
      else:
        args.append('--with-mpi-incdir="/usr/include"')  # dummy case

      if self.mpi.lib:
        args.append('--with-mpi-libdir="'+os.path.dirname(self.mpi.lib[0])+'"')
        libs = []
        for l in self.mpi.lib:
          ll = os.path.basename(l)
          if ll.endswith('.a'): libs.append(ll[3:-2])
          elif ll.endswith('.so'): libs.append(ll[3:-3])
          elif ll.endswith('.dylib'): libs.append(ll[3:-6])
          libs.append(ll[3:-2])
        libs = '-l' + ' -l'.join(libs)
        args.append('--with-mpi-libs="'+libs+'"')
      else:
        args.append('--with-mpi-libdir="/usr/lib"')  # dummy case
        args.append('--with-mpi-libs="-lc"')
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'zoltan'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('zoltan'):
      folder = os.path.join(self.packageDir, self.arch)
      if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
      os.mkdir(folder)
      try:
        self.logPrintBox('Configuring Zoltan; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && ../configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Zoltan: '+str(e))
      try:
        self.logPrintBox('Compiling Zoltan; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && make && make install && make clean ', timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on Zoltan: '+str(e))
      try:
        output3,err3,ret3  = PETSc.package.NewPackage.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(self.installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on Zoltan libraries: '+str(e))
      self.postInstall(output1+err1+output2+err2+output3+err3,'zoltan')
    return self.installDir
