'''
config.base.Configure is the base class for all configure objects. It handles several types of interaction:

Framework hooks
---------------

  The Framework will first instantiate the object and call setupDependencies(). All require()
  calls should be made in that method.

  The Framework will then call configure(). If it succeeds, the object will be marked as configured.

Generic test execution
----------------------

  All configure tests should be run using

  executeTest()

which formats the output and adds metadata for the log.

Preprocessing, Compiling, Linking, and Running
----------------------------------------------

  Two forms of this check are provided for each operation. The first is an "output" form which is
intended to provide the status and complete output of the command. The second, or "check" form will
return a success or failure indication based upon the status and output.

  outputPreprocess(), checkPreprocess(), preprocess()
  outputCompile(),    checkCompile()
  outputLink(),       checkLink()
  outputRun(),        checkRun()

  The language used for these operation is managed with a mask context:

  maskLanguage()

  We also provide special forms used to check for valid compiler and linker flags, optionally adding
them to the defaults.

  checkCompilerFlag(), addCompilerFlag()
  checkLinkerFlag(),   addLinkerFlag()

Finding Executables
-------------------

  getExecutable(), getExecutables(), checkExecutable()

Output
------

  addDefine(), addSubstitution(), addArgumentSubstitution(), addTypedef(), addPrototype()
  addMakeMacro(), addMakeRule()

  The object may define a headerPrefix member, which will be appended, followed
by an underscore, to every define which is output from it. Similarly, a substPrefix
can be defined which applies to every substitution from the object. Typedefs and
function prototypes are placed in a separate header in order to accomodate languges
such as Fortran whose preprocessor can sometimes fail at these statements.
'''
import logger
import script

import os
import time

class ConfigureSetupError(Exception):
  pass

class TmpDir(object):
  ''' a tmpdir keeps tracks of file names in a directory used for temporary
  tests, such as conftest files. 

  It is initialized with a base directory,

    b = TmpDir(base)

  To get the path,

    compilerDefines = b.join('confdefs.h') # compilerDefines = 'base/confdefs.h'

  One can optionally specify the threadSafe flag,

    b = TmpDir(base,threadSafe=True)

  and then every thread gets its own directory,

    compilerDefines = b.join('confdefs.h') # compilerDefines =
                                           # 'base/X/confdefs.h',
                                           # where X is unique to the current thread

  Using b.path('confdefs.h') will return an object that can be converted to a
  string at a later time (i.e., by another thread)
  '''

  from thread import get_ident as thread_id
  from threading import Lock
  
  _threadDir = {} # static table of the threads encountered, to convert their long id's to short numbers
  _threadCount = 0
  _lock = Lock()
  def __init__(self,base=None,infix='',threadSafe=None):
    if isinstance(base,TmpDir):
      self.threadSafe = base.threadSafe
      self.base = base.base
      self.infix = base.infix
    else:
      self.threadSafe = False
      self.base = str(base)
      self.infix = ''
    if infix:
      self.infix = os.path.join(self.infix,infix)
    if not threadSafe is None:
      self.threadSafe = threadSafe
    return

  def __str__(self):
    '''get the top of this thread's tmpDir for this config object'''
    return self.join('')

  def join(self,basename,mkdir=True):
    '''Immediately get the path to basename in this thread's tmpDir, i.e.,
    immediately convert to string'''
    if self.threadSafe:
      ident = thread_id()
      try:
        threadDir = TmpDir._threadDir[ident]
      except KeyError:
        with TmpDir._lock:
          threadDir = str(TmpDir._threadCount)
          TmpDir._threadCount += 1
        TmpDir._threadDir[ident] = threadDir
    else:
      threadDir = ''
    path = os.path.join(self.base,threadDir,self.infix,basename)
    if mkdir:
      if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path

  def path(self,basename):
    '''Return the path to basename in a hypothetical thread's tmpDir as an
    object that can be stringified by any thread at a later time'''
    if self.threadSafe:
      return TmpPath(self,basename)
    else:
      return self.join(basename)

class TmpPath(object):
  '''An object that can be converted to a path in any thread's tmpDir'''
  def __init__(self,tmpDir,basename):
    self.tmpDir = tmpDir
    self.basename = basename
  def __str__(self):
    return self.tmpDir.join(self.basename,mkdir=False)


class Configure(script.Script):
  def __init__(self, framework, tmpDir = None):
    script.Script.__init__(self, framework.clArgs, framework.argDB)
    self.framework       = framework
    self.defines         = {}
    self.makeRules       = {}
    self.makeMacros      = {}
    self.typedefs        = {}
    self.prototypes      = {}
    self.subst           = {}
    self.argSubst        = {}
    self.language        = 'C'
    if not tmpDir is None:
      self.tmpDir      = TmpDir(tmpDir)
    return

  def getTmpDir(self):
    if not hasattr(self, '_tmpDir'):
      self._tmpDir = TmpDir(base=self.framework.tmpDir,infix=self.__module__)
      if self._tmpDir.threadSafe:
        self.logPrint('All intermediate test results are stored in ' + os.path.join(self._tmpDir.base,'X',self._tmpDir.infix) + ' (X varies by thread)')
      else:
        self.logPrint('All intermediate test results are stored in ' + os.path.join(self._tmpDir.base,self._tmpDir.infix))
    return self._tmpDir
  def setTmpDir(self,temp):
    if hasattr(self, '_tmpDir'):
      if os.path.isdir(self._tmpDir.base):
        import shutil
        shutil.rmtree(self._tmpDir.base)
      if temp is None:
        delattr(self, '_tmpDir')
    if not temp is None:
      self._tmpDir = temp
    return
  tmpDir = property(getTmpDir, setTmpDir, doc = 'Temporary directory for test byproducts')

  def __str__(self):
    return ''

  def logError(self, component, status, output, error):
    if status:
      exitstr = ' exit code ' + str(status)
    else:
      exitstr = ''
    self.logWrite('Possible ERROR while running %s:%s\n' % (component, exitstr))
    if output:
      self.logWrite('stdout:\n' + output)
    if error:
      self.logWrite('stderr:\n' + error)

  def executeTest(self, test, args = [], kargs = {}):
    import time

    self.logWrite('================================================================================\n')
    self.logWrite('TEST '+str(test.im_func.func_name)+' from '+str(test.im_class.__module__)+'('+str(test.im_func.func_code.co_filename)+':'+str(test.im_func.func_code.co_firstlineno)+')\n')
    self.logPrint('TESTING: '+str(test.im_func.func_name)+' from '+str(test.im_class.__module__)+'('+str(test.im_func.func_code.co_filename)+':'+str(test.im_func.func_code.co_firstlineno)+')', debugSection = 'screen', indent = 0)
    if test.__doc__: self.logWrite('  '+test.__doc__+'\n')
    #t = time.time()
    if not isinstance(args, list): args = [args]
    ret = test(*args,**kargs)
    #self.logPrint('  TIME: '+str(time.time() - t)+' sec', debugSection = 'screen', indent = 0)
    return ret

  #################################
  # Define and Substitution Supported
  def addMakeRule(self, name, dependencies, rule = []):
    '''Designate that "name" should be rule in the makefile header (bmake file)'''
    self.logPrint('Defined make rule "'+name+'" with dependencies "'+str(dependencies)+'" and code '+str(rule))
    if not isinstance(rule,list): rule = [rule]
    self.makeRules[name] = [dependencies,rule]
    return

  def addMakeMacro(self, name, value):
    '''Designate that "name" should be defined to "value" in the makefile header (bmake file)'''
    self.logPrint('Defined make macro "'+name+'" to "'+str(value)+'"')
    self.makeMacros[name] = value
    return

  def getMakeMacro(self, name):
    return self.makeMacros.get(name)

  def delMakeMacro(self, name):
    '''Designate that "name" should be deleted (never put in) configuration header'''
    self.logPrint('Deleting "'+name+'"')
    if name in self.makeMacros: del self.makeMacros[name]
    return

  def addDefine(self, name, value):
    '''Designate that "name" should be defined to "value" in the configuration header'''
    self.logPrint('Defined "'+name+'" to "'+str(value)+'"')
    self.defines[name] = value
    return

  def delDefine(self, name):
    '''Designate that "name" should be deleted (never put in)  configuration header'''
    self.logPrint('Deleting "'+name+'"')
    if name in self.defines: del self.defines[name]
    return

  def addTypedef(self, name, value):
    '''Designate that "name" should be typedefed to "value" in the configuration header'''
    self.logPrint('Typedefed "'+name+'" to "'+str(value)+'"')
    self.typedefs[value] = name
    return

  def addPrototype(self, prototype, language = 'All'):
    '''Add a missing function prototype
       - The language argument defaults to "All"
       - Other language choices are C, Cxx, extern C'''
    self.logPrint('Added prototype '+prototype+' to language '+language)
    language = language.replace('+', 'x')
    if not language in self.prototypes:
      self.prototypes[language] = []
    self.prototypes[language].append(prototype)
    return

  def addSubstitution(self, name, value):
    '''Designate that "@name@" should be replaced by "value" in all files which experience substitution'''
    self.logPrint('Substituting "'+name+'" with "'+str(value)+'"')
    self.subst[name] = value
    return

  def addArgumentSubstitution(self, name, arg):
    '''Designate that "@name@" should be replaced by "arg" in all files which experience substitution'''
    self.logPrint('Substituting "'+name+'" with '+str(arg)+'('+str(self.argDB[arg])+')')
    self.argSubst[name] = arg
    return

  ################
  # Program Checks
  def checkExecutable(self, dir, name):
    prog  = os.path.join(dir, name)
    # also strip any \ before spaces, braces, so that we can specify paths the way we want them in makefiles.
    prog  = prog.replace('\ ',' ').replace('\(','(').replace('\)',')')
    found = 0
    self.logWrite('Checking for program '+prog+'...')
    if os.path.isfile(prog) and os.access(prog, os.X_OK):
      found = 1
      self.logWrite('found\n')
    else:
      self.logWrite('not found\n')
    return found

  def getExecutable(self, names, path = [], getFullPath = 0, useDefaultPath = 0, resultName = '', setMakeMacro = 1):
    '''Search for an executable in the list names
       - Each name in the list is tried for each entry in the path
       - If found, the path is stored in the variable "name", or "resultName" if given
       - By default, a make macro "resultName" will hold the path'''
    found = 0
    if isinstance(names, str):
      names = [names]
    if isinstance(path, str):
      path = path.split(os.path.pathsep)
    if not len(path):
      useDefaultPath = 1

    def getNames(name, resultName):
      import re
      prog = re.match(r'(.*?)(?<!\\)(\s.*)',name)
      if prog:
        name = prog.group(1)
        options = prog.group(2)
      else:
        options = ''
      if not resultName:
        varName = name
      else:
        varName = resultName
      return name, options, varName

    varName = names[0]
    varPath = ''
    for d in path:
      for name in names:
        name, options, varName = getNames(name, resultName)
        if self.checkExecutable(d, name):
          found = 1
          getFullPath = 1
          varPath = d
          break
      if found: break
    if useDefaultPath and not found:
      for d in os.environ['PATH'].split(os.path.pathsep):
        for name in names:
          name, options, varName = getNames(name, resultName)
          if self.checkExecutable(d, name):
            found = 1
            varPath = d
            break
        if found: break
    if not found:
      dirs = self.argDB['search-dirs']
      if not isinstance(dirs, list): dirs = [dirs]
      for d in dirs:
        for name in names:
          name, options, varName = getNames(name, resultName)
          if self.checkExecutable(d, name):
            found = 1
            getFullPath = 1
            varPath = d
            break
        if found: break

    if found:
      if getFullPath:
        setattr(self, varName, os.path.abspath(os.path.join(varPath, name))+options)
      else:
        setattr(self, varName, name+options)
      if setMakeMacro:
        self.addMakeMacro(varName.upper(), getattr(self, varName))
    return found

  def getExecutables(self, names, path = '', getFullPath = 0, useDefaultPath = 0, resultName = ''):
    '''Search for an executable in the list names
       - The full path given is searched for each name in turn
       - If found, the path is stored in the variable "name", or "resultName" if given'''
    for name in names:
      if self.getExecutable(name, path = path, getFullPath = getFullPath, useDefaultPath = useDefaultPath, resultName = resultName):
        return name
    return None

  class Mask(logger.Logger.Mask):
    '''To the Logger.Mask, we add the option of masking the
    config object's language at the same time as another attribute'''
    def __init__(self,target,name,val,maskLog = None,maskLanguage = None):
      logger.Logger.Mask.__init__(self,target,name,val,maskLog=maskLog)
      self.pushLanguage = maskLanguage
      self.pushLanguageMask = None
      return

    def __enter__(self):
      logger.Logger.Mask.__enter__(self)
      if self.pushLanguage:
        if self.pushLanguage == 'C++' or self.pushLanguage == 'CXX':
          self.pushLanguage = 'Cxx'
        self.pushLanguageMask = logger.Logger.Mask(self.target,'language',self.pushLanguage)
        self.pushLanguageMask.__enter__()
        self.target.logPrint('Pushing language '+self.pushLanguage)
      return

    def __exit__(self,exc_type,exc_value,traceback):
      if self.pushLanguageMask:
        self.target.logPrint('Popping language '+self.pushLanguage)
        self.pushLanguageMask.__exit__(exc_type, exc_value, traceback)
      logger.Logger.Mask.__exit__(self,exc_type, exc_value, traceback)
      return

  def mask(self,name,val,maskLog = None,maskLanguage = None):
    '''Mask an attribute for this thread only, should only be used in a
    with-statement: see the documentation for logger.Logger.Mask and
    logger.Logger.mask'''
    return self.Mask(self,name,val,maskLog=maskLog,maskLanguage=maskLanguage)

  def maskLanguage(self,lang,maskLog=None):
    '''Mask a config objects language'''
    return self.Mask(self,None,None,maskLog=maskLog,maskLanguage=lang)

  ###############################################
  # Preprocessor, Compiler, and Linker Operations
  def getHeaders(self):
    self.compilerDefines = self.tmpDir.path('confdefs.h')
    self.compilerFixes   = self.tmpDir.path('conffix.h')
    return

  def getPreprocessor(self):
    self.getHeaders()
    preprocessor       = self.framework.getPreprocessorObject(self.language)
    preprocessor.checkSetup()
    return preprocessor.getProcessor()

  def getCompiler(self):
    self.getHeaders()
    compiler            = self.framework.getCompilerObject(self.language)
    compiler.checkSetup()
    conftestbase = 'conftest'+compiler.sourceExtension
    self.compilerSource = self.tmpDir.path(conftestbase)
    self.compilerObj    = self.tmpDir.path(compiler.getTarget(conftestbase))
    return compiler.getProcessor()

  def getCompilerFlags(self):
    return self.framework.getCompilerObject(self.language).getFlags()

  def getLinker(self):
    self.getHeaders()
    linker            = self.framework.getLinkerObject(self.language)
    linker.checkSetup()
    conftestbase = 'conftest'+linker.sourceExtension
    self.linkerSource = self.tmpDir.path(conftestbase)
    self.linkerObj    = self.tmpDir.path(linker.getTarget(conftestbase, 0))
    return linker.getProcessor()

  def getLinkerFlags(self):
    return self.framework.getLinkerObject(self.language).getFlags()

  def getSharedLinker(self):
    self.getHeaders()
    linker            = self.framework.getSharedLinkerObject(self.language)
    linker.checkSetup()
    conftestbase = 'conftest'+linker.sourceExtension
    self.linkerSource = self.tmpDir.path(conftestbase)
    self.linkerObj    = self.tmpDir.path(linker.getTarget(conftestbase, 1))
    return linker.getProcessor()

  def getSharedLinkerFlags(self):
    return self.framework.getSharedLinkerObject(self.language).getFlags()

  def getDynamicLinker(self):
    self.getHeaders()
    linker            = self.framework.getDynamicLinkerObject(self.language)
    linker.checkSetup()
    conftestbase = 'conftest'+linker.sourceExtension
    self.linkerSource = self.tmpDir.path(conftestbase)
    self.linkerObj    = self.tmpDir.path(linker.getTarget(conftestbase, 1))
    return linker.getProcessor()

  def getDynamicLinkerFlags(self):
    return self.framework.getDynamicLinkerObject(self.language).getFlags()

  def getPreprocessorCmd(self):
    self.getCompiler()
    preprocessor = self.framework.getPreprocessorObject(self.language)
    preprocessor.checkSetup()
    preprocessor.includeDirectories.add(self.tmpDir)
    return preprocessor.getCommand(str(self.compilerSource))

  def getCompilerCmd(self):
    self.getCompiler()
    compiler = self.framework.getCompilerObject(self.language)
    compiler.checkSetup()
    compiler.includeDirectories.add(self.tmpDir)
    return compiler.getCommand(str(self.compilerSource), str(self.compilerObj))

  def getLinkerCmd(self):
    self.getLinker()
    linker = self.framework.getLinkerObject(self.language)
    linker.checkSetup()
    return linker.getCommand(str(self.linkerSource), str(self.linkerObj))

  def getFullLinkerCmd(self, objects, executable):
    self.getLinker()
    linker = self.framework.getLinkerObject(self.language)
    linker.checkSetup()
    return linker.getCommand(objects, executable)

  def getSharedLinkerCmd(self):
    self.getSharedLinker()
    linker = self.framework.getSharedLinkerObject(self.language)
    linker.checkSetup()
    return linker.getCommand(str(self.linkerSource), str(self.linkerObj))

  def getDynamicLinkerCmd(self):
    self.getDynamicLinker()
    linker = self.framework.getDynamicLinkerObject(self.language)
    linker.checkSetup()
    return linker.getCommand(str(self.linkerSource), str(self.linkerObj))

  def getCode(self, includes, body = None, codeBegin = None, codeEnd = None):
    language = self.language
    if includes and not includes[-1] == '\n':
      includes += '\n'
    if language in ['C', 'CUDA', 'Cxx']:
      codeStr = ''
      if str(self.compilerDefines): codeStr = '#include "'+os.path.basename(str(self.compilerDefines))+'"\n'
      codeStr += '#include "conffix.h"\n'+includes
      if not body is None:
        if codeBegin is None:
          codeBegin = '\nint main() {\n'
        if codeEnd is None:
          codeEnd   = ';\n  return 0;\n}\n'
        codeStr += codeBegin+body+codeEnd
    elif language == 'FC':
      if not includes is None:
        codeStr = includes
      else:
        codeStr = ''
      if not body is None:
        if codeBegin is None:
          codeBegin = '      program main\n'
        if codeEnd is None:
          codeEnd   = '\n      end\n'
        codeStr += codeBegin+body+codeEnd
    else:
      raise RuntimeError('Cannot determine code body for language: '+language)
    return codeStr

  def preprocess(self, codeStr, timeout = 600.0):
    def report(command, status, output, error):
      if error or status:
        self.logError('preprocessor', status, output, error)
        self.logWrite('Source:\n'+self.getCode(codeStr))

    command = self.getPreprocessorCmd()
    if str(self.compilerDefines): self.framework.outputHeader(str(self.compilerDefines))
    self.framework.outputCHeader(str(self.compilerFixes))
    f = file(str(self.compilerSource), 'w')
    f.write(self.getCode(codeStr))
    f.close()
    (out, err, ret) = Configure.executeShellCommand(command, checkCommand = report, timeout = timeout, log = self.log, lineLimit = 100000)
    if self.cleanup:
      for filename in [str(self.compilerDefines), str(self.compilerFixes), str(self.compilerSource)]:
        if os.path.isfile(filename):
          os.remove(filename)
    return (out, err, ret)

  def outputPreprocess(self, codeStr):
    '''Return the contents of stdout when preprocessing "codeStr"'''
    self.logWrite('Source:\n'+self.getCode(codeStr))
    return self.preprocess(codeStr)[0]

  def checkPreprocess(self, codeStr, timeout = 600.0):
    '''Return True if no error occurred
       - An error is signaled by a nonzero return code, or output on stderr'''
    (out, err, ret) = self.preprocess(codeStr, timeout = timeout)
    #pgi dumps filename on stderr - but returns 0 errorcode'
    if err =='conftest.c:': err = ''
    err = self.framework.filterPreprocessOutput(err, self.log)
    return not ret and not len(err)

  # Should be static
  def getPreprocessorFlagsName(self, language):
    if language in ['C', 'Cxx', 'FC']:
      flagsArg = 'CPPFLAGS'
    elif language == 'CUDA':
      flagsArg = 'CUDAPPFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg

  def getPreprocessorFlagsArg(self):
    '''Return the name of the argument which holds the preprocessor flags for the current language'''
    return self.getPreprocessorFlagsName(self.language)

  def filterCompileOutput(self, output):
    return self.framework.filterCompileOutput(output)

  def outputCompile(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    '''Return the error output from this compile and the return code'''
    def report(command, status, output, error):
      if error or status:
        self.logError('compiler', status, output, error)
      else:
        self.logWrite('Successful compile:\n')
      self.logWrite('Source:\n'+self.getCode(includes, body, codeBegin, codeEnd))

    cleanup = cleanup and self.framework.doCleanup
    command = self.getCompilerCmd()
    if str(self.compilerDefines): self.framework.outputHeader(str(self.compilerDefines))
    self.framework.outputCHeader(str(self.compilerFixes))
    f = file(str(self.compilerSource), 'w')
    f.write(self.getCode(includes, body, codeBegin, codeEnd))
    f.close()
    (out, err, ret) = Configure.executeShellCommand(command, checkCommand = report, log = self.log)
    if not os.path.isfile(str(self.compilerObj)):
      err += '\nPETSc Error: No output file produced'
    if cleanup:
      for filename in [str(self.compilerDefines), str(self.compilerFixes), str(self.compilerSource), str(self.compilerObj)]:
        if os.path.isfile(filename):
          os.remove(filename)
    return (out, err, ret)

  def checkCompile(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    '''Returns True if the compile was successful'''
    (output, error, returnCode) = self.outputCompile(includes, body, cleanup, codeBegin, codeEnd)
    output = self.filterCompileOutput(output+'\n'+error)
    return not (returnCode or len(output))

  # Should be static
  def getCompilerFlagsName(self, language, compilerOnly = 0):
    if language == 'C':
      flagsArg = 'CFLAGS'
    elif language == 'CUDA':
      flagsArg = 'CUDAFLAGS'
    elif language == 'Cxx':
      if compilerOnly:
        flagsArg = 'CXX_CXXFLAGS'
      else:
        flagsArg = 'CXXFLAGS'
    elif language == 'FC':
      flagsArg = 'FFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg

  def getCompilerFlagsArg(self, compilerOnly = 0):
    '''Return the name of the argument which holds the compiler flags for the current language'''
    return self.getCompilerFlagsName(self.language, compilerOnly)

  def filterLinkOutput(self, output):
    return self.framework.filterLinkOutput(output)

  def outputLink(self, includes, body, cleanup = 1, codeBegin = None, codeEnd = None, shared = 0, linkLanguage=None, examineOutput=lambda ret,out,err:None):
    import sys

    (out, err, ret) = self.outputCompile(includes, body, cleanup = 0, codeBegin = codeBegin, codeEnd = codeEnd)
    examineOutput(ret, out, err)
    out = self.filterCompileOutput(out+'\n'+err)
    if ret or len(out):
      self.logPrint('Compile failed inside link\n'+out)
      self.linkerObj = ''
      return (out, ret)

    cleanup = cleanup and self.framework.doCleanup

    if not linkLanguage:
      linkLanguage = self.language
    with self.maskLanguage(linkLanguage):
      if shared == 'dynamic':
        cmd = self.getDynamicLinkerCmd()
      elif shared:
        cmd = self.getSharedLinkerCmd()
      else:
        cmd = self.getLinkerCmd()

    linkerObj = self.linkerObj
    def report(command, status, output, error):
      if error or status:
        self.logError('linker', status, output, error)
        examineOutput(status, output, error)
      return
    (out, err, ret) = Configure.executeShellCommand(cmd, checkCommand = report, log = self.log)
    self.linkerObj = linkerObj
    if os.path.isfile(str(self.compilerObj)):
      import threading
      os.remove(str(self.compilerObj))
    if cleanup:
      if os.path.isfile(str(self.linkerObj)):
        os.remove(str(self.linkerObj))
      pdbfile = os.path.splitext(str(self.linkerObj))[0]+'.pdb'
      if os.path.isfile(pdbfile):
        os.remove(pdbfile)
    return (out+'\n'+err, ret)

  def checkLink(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None, shared = 0, linkLanguage=None, examineOutput=lambda ret,out,err:None):
    (output, returnCode) = self.outputLink(includes, body, cleanup, codeBegin, codeEnd, shared, linkLanguage, examineOutput)
    output = self.filterLinkOutput(output)
    return not (returnCode or len(output))

  # Should be static
  def getLinkerFlagsName(self, language):
    if language in ['C', 'CUDA', 'Cxx', 'FC']:
      flagsArg = 'LDFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg

  def getLinkerFlagsArg(self):
    '''Return the name of the argument which holds the linker flags for the current language'''
    return self.getLinkerFlagsName(self.language)

  def outputRun(self, includes, body, cleanup = 1, defaultOutputArg = '', executor = None):
    if not self.checkLink(includes, body, cleanup = 0): return ('', 1)
    self.logWrite('Testing executable '+str(self.linkerObj)+' to see if it can be run\n')
    if not os.path.isfile(str(self.linkerObj)):
      self.logWrite('ERROR executable '+str(self.linkerObj)+' does not exist\n')
      return ('', 1)
    if not os.access(str(self.linkerObj), os.X_OK):
      self.logWrite('ERROR while running executable: '+str(self.linkerObj)+' is not executable\n')
      return ('', 1)
    if self.argDB['with-batch']:
      if defaultOutputArg:
        if defaultOutputArg in self.argDB:
          return (self.argDB[defaultOutputArg], 0)
        else:
          raise ConfigureSetupError('Must give a default value for '+defaultOutputArg+' since executables cannot be run')
      else:
        raise ConfigureSetupError('Running executables on this system is not supported')
    cleanup = cleanup and self.framework.doCleanup
    if executor:
      command = executor+' '+str(self.linkerObj)
    else:
      command = str(self.linkerObj)
    output  = ''
    error   = ''
    status  = 1
    self.logWrite('Executing: '+command+'\n')
    try:
      (output, error, status) = Configure.executeShellCommand(command, log = self.log)
    except RuntimeError, e:
      self.logWrite('ERROR while running executable: '+str(e)+'\n')
    if os.path.isfile(str(self.compilerObj)):
      try:
        os.remove(str(self.compilerObj))
      except RuntimeError, e:
        self.logWrite('ERROR while removing object file: '+str(e)+'\n')
    if cleanup and os.path.isfile(str(self.linkerObj)):
      try:
        if os.path.exists('/usr/bin/cygcheck.exe'): time.sleep(1)
        os.remove(str(self.linkerObj))
      except RuntimeError, e:
        self.logWrite('ERROR while removing executable file: '+str(e)+'\n')
    return (output+error, status)

  def checkRun(self, includes = '', body = '', cleanup = 1, defaultArg = '', executor = None):
    (output, returnCode) = self.outputRun(includes, body, cleanup, defaultArg, executor)
    return not returnCode

  def splitLibs(self,libArgs):
    '''Takes a string containing a list of libraries (including potentially -L, -l, -w etc) and generates a list of libraries'''
    dirs = []
    libs = []
    for arg in libArgs.split(' '):
      if not arg: continue
      if arg.startswith('-L'):
        dirs.append(arg[2:])
      elif arg.startswith('-l'):
        libs.append(arg[2:])
      elif not arg.startswith('-'):
        libs.append(arg)
    libArgs = []
    for lib in libs:
      if not os.path.isabs(lib):
        added = 0
        for dir in dirs:
          if added:
	    break
	  for ext in ['a', 'so','dylib']:
            filename = os.path.join(dir, 'lib'+lib+'.'+ext)
            if os.path.isfile(filename):
              libArgs.append(filename)
              added = 1
              break
      else:
        libArgs.append(lib)
    return libArgs

  def splitIncludes(self,incArgs):
    '''Takes a string containing a list of include directories with -I and generates a list of includes'''
    includes = []
    for inc in incArgs.split(' '):
      if inc.startswith('-I'):
        # check if directory exists?
        includes.append(inc[2:])
    return includes

  def setupPackageDependencies(self, framework):
    '''All calls to the framework addPackageDependency() should be made here'''
    pass

  def setupDependencies(self, framework):
    '''All calls to the framework require() should be made here'''
    self.framework = framework

  def configure(self):
    pass

  def no_configure(self):
    pass
