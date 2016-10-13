#!/usr/bin/env python

import os
from distutils.sysconfig import parse_makefile
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from cmakegen import Mistakes, stripsplit, AUTODIRS, SKIPDIRS
from cmakegen import defaultdict # collections.defaultdict, with fallback for python-2.4
from gmakegen import *


import inspect
thisscriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
maintdir=os.path.join(os.path.join(os.path.dirname(thisscriptdir),'bin'),'maint')
sys.path.insert(0,maintdir) 
from examplesWalker import *

class generateExamples(Petsc,PETScExamples):
  """
  Why dual inheritance:
    gmakegen.py has basic structure for finding the files, writing out
      the dependencies, etc.
     exampleWalker has the logic for migrating the tests, analyzing the tests, etc.
     Rather than use the os.walk from gmakegen, gmakegentest re-uses the
     exampleWalker functionality
  """
  def __init__(self,petsc_dir=None, petsc_arch=None, verbose=False, single_ex=False):
    super(generateExamples, self).__init__(petsc_dir=None, petsc_arch=None, verbose=False)

    self.single_ex=single_ex
    self.arch_dir=os.path.join(self.petsc_dir,self.petsc_arch)
    self.ptNaming=True
    # Whether to write out a useful debugging
    #if verbose: self.summarize=True
    self.summarize=True

    # For help in setting the requirements
    self.precision_types="single double quad int32".split()
    self.integer_types="int32 int64".split()
    self.languages="fortran cuda cxx".split()    # Always requires C so do not list

    # Adding a dictionary for storing sources, objects, and tests
    # to make building the dependency tree easier
    self.sources={}
    self.objects={}
    self.tests={}
    for pkg in PKGS:
      self.sources[pkg]={}
      self.objects[pkg]=[]
      self.tests[pkg]={}
      for lang in LANGS:
        self.sources[pkg][lang]=[]
        self.tests[pkg][lang]=[]

    # Save the examplate template in a string
    template_name=os.path.join(thisscriptdir,"example_template.sh.in")
    fh=open(template_name,"r")
    self.templateStr=fh.read()
    fh.close

    # Get variables that are needed by the scripts
    self.varSubst={}
    self.varSubst['MPIEXEC']=self.conf['MPIEXEC']
    self.varSubst['DIFF']=self.conf['DIFF']
    self.varSubst['RM']=self.conf['RM']
    self.varSubst['GREP']=self.conf['GREP']

    return

  def parseExampleFile(self,srcfile,basedir,srcDict):
    """
    Parse the example files and store the relevant information into a
    dictionary to process later
    """
    curdir=os.path.realpath(os.path.curdir)
    os.chdir(basedir)

    basename=os.path.splitext(srcfile)[0]

    sh=open(srcfile,"r"); fileStr=sh.read(); sh.close()
    fsplit=fileStr.split("/*TEST")[1:]
    if len(fsplit)==0: return False
    srcTests=[]
    for t in fsplit: srcTests.append(t.split("TEST*/")[0].strip())

    # Now take the strings and put them into a dictionary
    for test in srcTests:

      # Required strings: script OR nsize
      foundSize=False; foundScript=False

      # The dictionary key is determined by output_suffix.
      # Allow anywhere in the file so need to grab it first
      testname="run"+basename
      if os.path.splitext(srcfile)[1].lstrip(".").startswith("F"):
        test=test.replace("!","")
      for line in test.split("\n"):
        if not ":" in line: continue  # This shouldn't happen
        var=line.split(":")[0].strip()
        val=line.split(":")[1].strip()
        if var=="output_suffix":
          if len(val)>0:
            testname=testname+"_"+val
        if var=="nsize": foundSize=True
        if var=="script": foundScript=True

      #print "1> ", self.nameSpace(srcfile,basedir), testname
      # If folks keep forgetting to put output_suffix, problems
      rpath=self.relpath(curdir, basedir)
      if srcDict.has_key(testname):
        print "Duplicate test in file "+srcfile+" in "+rpath +". Check output_suffix."
        continue

      # Make sure it passes basic requirements
      if not (foundSize or foundScript):
        print "Test with insufficient info in file "+srcfile+" in "+rpath +". Check nsize."
        continue

      srcDict[testname]={}
      if foundSize: 
        srcDict[testname]['abstracted']=True
      else:
        srcDict[testname]['abstracted']=False
      
      i=-1
      for line in test.split("\n"):
        if not ":" in line: continue  # This shouldn't happen
        i=i+1
        var=line.split(":")[0].strip()
        val=line.split(":")[1].strip()
        # requires is comma delimited list so make list
        if var=="nsize": val=int(val) 
        if var=="requires": 
          val=val.split(",")
        # do assume that script is the last entry
        # so that parsing doesn't cause problems
        if var=="script": val=test.split("script:")[1].strip()

        srcDict[testname][var]=val
        if var=="script": break
         
    os.chdir(curdir)
    return True

  def getLanguage(self,srcfile):
    """
    Based on the source, determine associated language as found in gmakegen.LANGS
    Can we just return srcext[1:\] now?
    """
    langReq=None
    srcext=os.path.splitext(srcfile)[-1]
    if srcext in ".F90".split(): langReq="F90"
    if srcext in ".F".split(): langReq="F"
    if srcext in ".cxx".split(): langReq="cxx"
    if srcext == ".cu": langReq="cu"
    if srcext == ".c": langReq="c"
    if not langReq: print "ERROR: ", srcext, srcfile
    return langReq

  def addToSources(self,exfile,root):
    """
      Put into data structure that allows easy generation of makefile
    """
    pkg=self.relpath(self.petsc_dir,root).split("/")[1]
    fullfile=os.path.join(root,exfile)
    relpfile=self.relpath(self.petsc_dir,fullfile)
    self.sources[pkg][self.getLanguage(exfile)].append(relpfile)

    # In gmakefile, ${TESTDIR} var specifies the object compilation
    testsdir=self.relpath(self.petsc_dir,root)+"/"
    objfile="${TESTDIR}/"+testsdir+os.path.splitext(exfile)[0]+".o"
    self.objects[pkg].append(objfile)
    return

  def addToTests(self,test,root,exfile):
    """
      Put into data structure that allows easy generation of makefile
      Organized by languages to allow testing of languages
    """
    pkg=self.relpath(self.petsc_dir,root).split("/")[1]
    nmtest=self.nameSpace(test,root)
    self.tests[pkg][self.getLanguage(exfile)].append(nmtest)
    return

  def getExecname(self,exfile,root):
    """
      Generate bash script using template found next to this file.  
      This file is read in at constructor time to avoid file I/O
    """
    rpath=self.relpath(self.petsc_dir,root)
    if self.single_ex:
      execname=rpath.split("/")[1]+"-ex"
    else:
      execname=os.path.splitext(exfile)[0]
    return execname


  def genRunScript(self,testname,root,testDict):
    """
      Generate bash script using template found next to this file.  
      This file is read in at constructor time to avoid file I/O
    """
    # runscript_dir directory has to be consistent with gmakefile
    rpath=self.relpath(self.petsc_dir,root)
    runscript_dir=os.path.join(self.arch_dir,"tests",rpath)
    if not os.path.isdir(runscript_dir): os.makedirs(runscript_dir)
    fh=open(os.path.join(runscript_dir,testname+".sh"),"w")
    petscvarfile=os.path.join(self.arch_dir,'lib','petsc','conf','petscvariables')
    
    subst={}
    if testDict['abstracted']:
      # Setup the variables in template_string that need to be substituted
      subst['SRCDIR']=os.path.join(self.petsc_dir,rpath)
      subst['MPIARGSIZE']="-n "+str(testDict['nsize'])
      subst['EXEC']="../"+testDict['execname']
      subst['ARGS']=(testDict['args'] if testDict.has_key('args') else " ")
      subst['TESTNAME']=testname
      outf=testname+".out"
      subst['OUTPUTFILE']=(re.sub("runnew_","",outf) if outf.startswith("runnew_") else outf)
      testStr=self.templateStr
    else:
      testStr=testDict['script']
      # Try to retrofit the existing scripts (bit of hoop jumping
      exname=testDict['execname']
      patname=(re.sub("new_","",exname) if exname.startswith("new_") else exname)
      patt=" "+patname+" ";   subs=" ../"+exname+" ";    testStr=re.sub(patt,subs,testStr)
      patt=" ./"+patname+" "; subs=" ../"+exname+" ";    testStr=re.sub(patt,subs,testStr)
      patt=" output/"      ; subs=" "+root+"/output/";  testStr=re.sub(patt,subs,testStr)

    # Now substitute the key variables
    allVars=self.varSubst.copy(); allVars.update(subst)
    for subkey in allVars:
      patt="\${"+subkey+"}"
      testStr=re.sub(patt,allVars[subkey],testStr)

    fh.write(testStr+"\n"); fh.close()
    os.chmod(os.path.join(runscript_dir,testname+".sh"),0777)
    return

  def  genScriptsAndInfo(self,exfile,root,srcDict):
    """
    For every test in the exfile with info in the srcDict:
      1. Determine if it needs to be run for this arch
      2. Generate the script
      3. Generate the data needed to write out the makefile in a
         convenient way
    """
    debug=False
    fileIsTested=False
    execname=self.getExecname(exfile,root)
    if self._isBuilt(exfile):
      for test in srcDict:
        srcDict[test]['execname']=execname   # Convenience in generating scripts
        srcDict[test]['isrun']=False
        if self._isRun(srcDict[test]):
          fileIsTested=True
          self.genRunScript(test,root,srcDict[test])
          srcDict[test]['isrun']=True        # Convenience for debugging
          self.addToTests(test,root,exfile)
        if debug: print self.nameSpace(exfile,root), test

    # This adds to datastructure for building deps
    if fileIsTested: self.addToSources(exfile,root)
    return

  def _isBuilt(self,exfile):
    """
    Determine if this file should be built. 
    """
    # Get the language based on file extension
    lang=self.getLanguage(exfile)
    if lang=="f" and not self.have_fortran: return False
    if lang=="cu" and not self.conf.has_key('PETSC_HAVE_CUDA'): return False
    if lang=="cxx" and not self.conf.has_key('PETSC_HAVE_CXX'): return False
    return True


  def _isRun(self,testDict):
    """
    Based on the requirements listed in the src file and the petscconf.h
    info, determine whether this test should be run or not.
    """
    indent="  "
    debug=False

    # MPI requirements
    if testDict.has_key('nsize'):
      if testDict['nsize']>1 and self.conf['MPI_IS_MPIUNI']==1: 
        if debug: print indent+"Cannot run parallel tests"
        return False
    else:
      # If we don't know nsize, then assume it cannot be run
      if self.conf['MPI_IS_MPIUNI']==1: return False
 
    if testDict.has_key('requires'):
      for requirement in testDict['requires']:
        requirement=requirement.strip()
        if not requirement: continue
        if debug: print indent+"Requirement: ", requirement
        isNull=False
        if requirement.startswith("!"):
          requirement=requirement[1:]; isNull=True
        # Scalar requirement
        if requirement=="complex":
          if self.conf['PETSC_SCALAR']=='complex':
            if isNull: return False
          else:
            return False
        # Precision requirement for reals
        if requirement in self.precision_types:
          if self.conf['PETSC_PRECISION']==requirement:
            if isNull: return False
          else:
            return False
        # Precision requirement for ints
        if requirement in self.integer_types:
          if requirement=="int32":
            if self.conf['PETSC_SIZEOF_INT']==4:
              if isNull: return False
            else:
              return False
          if requirement=="int64":
            if self.conf['PETSC_SIZEOF_INT']==8:
              if isNull: return False
            else:
              return False
        # Defines
        if "define(" in requirement:
          reqdef=requirement.split("(")[1].split(")")[0]
          val=(reqdef.split()[1] if " " in reqdef else "")
          if self.conf.has_key(reqdef):
            if val:
              if self.conf[reqdef]==val:
                if isNull: return False
              else:
                return False
            else:
              if isNull: return False
          else:
            return False

        # Rest should be packages that we can just get from conf
        petscconfvar="PETSC_HAVE_"+requirement.upper()
        if self.conf.get(petscconfvar):
          if isNull: return False
        else:
          if debug: print "requirement not found: ", requirement
          return False

    return True

  def genPetscTests_summarize(self,dataDict):
    """
    Required method to state what happened
    """
    if not self.summarize: return
    indent="   "
    fhname="GenPetscTests_summarize.txt"
    fh=open(fhname,"w")
    print "See ", fhname
    for root in dataDict:
      relroot=self.relpath(self.petsc_dir,root)
      pkg=relroot.split("/")[1]
      fh.write(relroot+"\n")
      allSrcs=[]
      for lang in LANGS: allSrcs=allSrcs+self.sources[pkg][lang]
      for exfile in dataDict[root]:
        # Basic  information
        fullfile=os.path.join(root,exfile)
        rfile=self.relpath(self.petsc_dir,fullfile)
        builtStatus=(" Is built" if rfile in allSrcs else " Is NOT built")
        fh.write(indent+exfile+indent*4+builtStatus+"\n")

        for test in dataDict[root][exfile]:
          line=indent*2+test
          fh.write(line+"\n")
          # Looks nice to have the keys in order
          #for key in dataDict[root][exfile][test]:
          for key in "isrun abstracted nsize args requires script".split():
            if not dataDict[root][exfile][test].has_key(key): continue
            line=indent*3+key+": "+str(dataDict[root][exfile][test][key])
            fh.write(line+"\n")
          fh.write("\n")
        fh.write("\n")
      fh.write("\n")
    #fh.write("\nClass Sources\n"+str(self.sources)+"\n")
    #fh.write("\nClass Tests\n"+str(self.tests)+"\n")
    fh.close()
    return

  def genPetscTests(self,root,dirs,files,dataDict):
    """
     Go through and parse the source files in the directory to generate
     the examples based on the metadata contained in the source files
    """
    debug=False
    # Use examplesAnalyze to get what the makefles think are sources
    #self.examplesAnalyze(root,dirs,files,anlzDict)

    dataDict[root]={}

    for exfile in files:
      #TST: Until we replace files, still leaving the orginals as is
      if not exfile.startswith("new_"+"ex"): continue
      dataDict[root][exfile]={}
      self.parseExampleFile(exfile,root,dataDict[root][exfile])
      self.genScriptsAndInfo(exfile,root,dataDict[root][exfile])

    return

  def gen_pkg(self, pkg):
    """
     Overwrite of the method in the base PETSc class 
    """
    return self.sources[pkg]

  def write_gnumake(self,dataDict):
    """
     Write out something similar to files from gmakegen.py

     There is not a lot of has_key type checking because
     should just work and need to know if there are bugs

     Test depends on script which also depends on source
     file, but since I don't have a good way generating
     acting on a single file (oops) just depend on
     executable which in turn will depend on src file
    """
    # Open file
    arch_files = self.arch_path('lib','petsc','conf', 'testfiles')
    fd = open(arch_files, 'w')

    # Write out the sources
    gendeps = self.gen_gnumake(fd,prefix="testsrcs-")

    # Write out the tests and execname targets
    fd.write("\n#Tests and executables\n")    # Delimiter
    testdeps=" ".join(["test-"+pkg for pkg in PKGS])
    fd.write("test: "+testdeps+"\n")    # Main test target

    for pkg in PKGS:
      # These grab the ones that are built
      # Package tests
      testdeps=" ".join(["test-"+pkg+"-"+lang for lang in LANGS])
      fd.write("test-"+pkg+": "+testdeps+"\n")
      if self.single_ex:
        execname=pkg+"-ex"
        fd.write(execname+": "+" ".join(self.objects[pkg])+"\n\n")
      for lang in LANGS:
        testdeps=" ".join(self.tests[pkg][lang])
        fd.write("test-"+pkg+"-"+lang+":"+testdeps+"\n")
        for exfile in self.sources[pkg][lang]:
          filetests=[]
          root=os.path.join(self.petsc_dir,os.path.dirname(exfile))
          basedir=os.path.dirname(exfile)
          testdir="${TESTDIR}/"+basedir+"/"
          base=os.path.basename(exfile)
          objfile=testdir+os.path.splitext(base)[0]+".o"
          linker=self.getLanguage(exfile)[0].upper()+"LINKER"
          if not self.single_ex:
            localexec=os.path.basename(os.path.splitext(exfile)[0])
            execname=os.path.join(testdir,localexec)
            localobj=os.path.basename(objfile)
            fd.write("\n"+execname+": "+objfile+" ${libpetscall}\n")
            # There should be a better way here
            line="\t-cd "+testdir+"; ${"+linker+"} -o "+localexec+" "+localobj+" ${PETSC_LIB}"
            fd.write(line+"\n")
          for test in dataDict[root][base]:
            if dataDict[root][base][test]['isrun']:
              fulltest=self.nameSpace(test,root)
              filetests.append(fulltest)
              script=test+".sh"
              fd.write(fulltest+": "+execname+"\n")
              rundir=os.path.join(testdir,test)
              cmd="mkdir -p "+rundir+"; cd "+rundir+"; ../"+script
              fd.write("\t-@"+cmd+"\n")
          linker=self.getLanguage(exfile)[0].upper()+"LINKER"
          allFileTestsTarg=self.nameSpace(os.path.splitext(base)[0],root)+"-all"
          fd.write(allFileTestsTarg+": "+" ".join(filetests)+"\n")

    fd.write("helptests:\n\t -@grep '^[a-z]' ${generatedtest} | cut -f1 -d:\n")
    # Write out tests
    return

  def writeHarness(self,output,dataDict):
    """
     This is set up to write out multiple harness even if only gnumake
     is supported now
    """
    eval("self.write_"+output+"(dataDict)")
    return

def main(petsc_dir=None, petsc_arch=None, output=None, verbose=False, single_ex=False):
    if output is None:
        output = 'gnumake'
    pEx=generateExamples(petsc_dir=petsc_dir, petsc_arch=petsc_arch, verbose=verbose, single_ex=single_ex)
    dataDict=pEx.walktree(os.path.join(pEx.petsc_dir,'src'),action="genPetscTests")
    pEx.writeHarness(output,dataDict)

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--verbose', help='Show mismatches between makefiles and the filesystem', action='store_true', default=False)
    parser.add_option('--petsc-arch', help='Set PETSC_ARCH different from environment', default=os.environ.get('PETSC_ARCH'))
    parser.add_option('--output', help='Location to write output file', default=None)
    parser.add_option('-s', '--singe_executable', dest='single_executable', action="store_false", help='Whether there should be single executable per src subdir.  Default is false')
    opts, extra_args = parser.parse_args()
    if extra_args:
        import sys
        sys.stderr.write('Unknown arguments: %s\n' % ' '.join(extra_args))
        exit(1)
    main(petsc_arch=opts.petsc_arch, output=opts.output, verbose=opts.verbose, single_ex=opts.single_executable)
