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

    # Language for requirements
    self.precision_types="single double quad int32".split()
    self.integer_types="int32 int64".split()
    self.languages="fortran cuda cxx".split()    # Always requires C so do not list

    # Save the examplate template in a string
    template_name=os.path.join(thisscriptdir,"example_template.sh.in")
    fh=open(template_name,"r")
    self.templateStr=fh.read()
    fh.close

    # Adding a dictionary for storing sources
    self.sources={}
    for pkg in PKGS:
      self.sources[pkg]=[]

    self.ptNaming=True
    return

  def parseExampleFile(self,srcfile,basedir,srcDict):
    """
    Parse the example files and store the relevant information into a
    dictionary to process later
    """
    curdir=os.path.realpath(os.path.curdir)
    os.chdir(basedir)

    basename=os.path.splitext(srcfile)[0]
    srcext=os.path.splitext(srcfile)[-1]
    langReq=""
    if srcext in "F F90 f f90".split(): langReq="fortran"
    if srcext in "cu".split(): langReq="cuda"
    if srcext in "cxx".split(): langReq="cxx"

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
          if langReq: val.append(langReq)
        # do assume that script is the last entry
        # so that parsing doesn't cause problems
        if var=="script": val=test.split("script:")[1].strip()

        srcDict[testname][var]=val
        if var=="script": break
         
    os.chdir(curdir)
    return True

  def addToSources(self,exfile,root):
    """
      Put into data structure that allows easy generation of makefile
      Note that our testfiles are not separated by language
    """
    pkg=self.relpath(self.petsc_dir,root).split("/")[1]
    self.sources[pkg].append(exfile)
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
    runscript_dir=os.path.join(self.petsc_dir,self.petsc_arch,"tests",rpath.lstrip("src/"))
    if not os.path.isdir(runscript_dir): os.makedirs(runscript_dir)
    fh=open(os.path.join(runscript_dir,testname+".sh"),"w")
    
    if testDict['abstracted']:
      # Setup the variables in template_string that need to be substituted
      subst={}
      subst['MPIEXEC']=self.conf['MPIEXEC']
      subst['MPIARGSIZE']="-n "+str(testDict['nsize'])
      subst['EXEC']=testDict['execname']
      subst['ARGS']=(testDict['args'] if testDict.has_key('args') else " ")
      subst['TESTNAME']=testname
      subst['SRCDIR']=os.path.join(self.petsc_dir,'src')
      subst['DIFF']=self.conf['DIFF']
      subst['RM']=self.conf['RM']
      testStr=self.templateStr
      for subkey in subst:
        patt="\${"+subkey+"}"
        print  patt, subst[subkey]
        testStr=re.sub(patt,subst[subkey],testStr)
      print root, testname, testStr
      sys.exit()
      fh.write(testStr+"\n")
    else:
      fh.write(testDict['script']+"\n")

    fh.close()
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
    for test in srcDict:
      srcDict[test]['execname']=execname  # Convenience
      isrun=self.determineIfRun(srcDict[test])
      if isrun:
        fileIsTested=True
        self.genRunScript(test,root,srcDict[test])
      if debug: print self.nameSpace(exfile,root), test, isrun
    if fileIsTested:
      self.addToSources(exfile,root)
    return

  def determineIfRun(self,testDict):
    """
    Based on the requirements listed in the src file and the petscconf.h
    info, determine whether this test should be run or not.
    """
    isrun=True
    indent="  "
    debug=False

    # MPI requirements
    if testDict.has_key('nsize'):
      if testDict['nsize']>1 and self.conf['MPI_IS_MPIUNI']==1: 
        if debug: print indent+"Cannot run parallel tests"
        return False
 
    if testDict.has_key('requires'):
      for requirement in testDict['requires']:
        if debug: print indent+"Requirement: ", requirement
        isNull=False
        if requirement.startswith("!"):
          requirement=requirement[1:]; isNull=True
        # Language requirement
        if requirement in self.languages:
          if self.conf['PETSC_LANGUAGE']:
            pass # To Do
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
        if self.conf.get(requirement): 
          if isNull: return False

    return isrun

  def genPetscTests_summarize(self,dataDict):
    """
    Required method to state what happened
    """
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


def main(petsc_dir=None, petsc_arch=None, output=None, verbose=False, single_ex=False):
    if output is None:
        output = 'gnumake'
    pEx=generateExamples(petsc_dir=petsc_dir, petsc_arch=petsc_arch, verbose=verbose, single_ex=single_ex)
    startdir=os.path.realpath(os.path.curdir)
    pEx.walktree(startdir,action="genPetscTests")

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
