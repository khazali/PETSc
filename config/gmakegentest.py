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
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
maintdir=os.path.join(os.path.join(os.path.dirname(currentdir),'bin'),'maint')
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
    self.languages="fortran cuda cxx".split()    # Always requires C so do not list
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
    for t in fsplit[1:]: 
      srcTests.append(t.split("TEST*/")[0].strip())

    # Now take the strings and put them into a dictionary
    for test in srcTests:

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

      # If folks keep forgetting to put output_suffix, problems
      if srcDict.has_key(testname):
        print "Duplicate test name detected in file "+srcfile+" in "+basedir
        print "  Check output_suffix in tests"
        continue
      srcDict[testname]={}
      
      i=-1
      for line in test.split("\n"):
        if not ":" in line: continue  # This shouldn't happen
        i=i+1
        var=line.split(":")[0].strip()
        val=line.split(":")[1].strip()
        # requires is comma delimited list so make list
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

  def  genRunScript(self,exfile,root,srcDict):
    """
      Generate the bash script
    """
    return

  def  genScriptsAndInfo(self,exfile,root,srcDict):
    """
    For every test in the exfile with info in the srcDict:
      1. Determine if it needs to be run for this arch
      2. Generate the script
      3. Generate the data needed to write out the makefile in a
         convenient way
    """
    for test in srcDict:
      isrun=self.determineIfRun(test,srcDict[test])
      if isrun:
        self.genRunScript(test,root,srcDict[test])
    return

  def determineIfRun(self,testName,testDict):
    """
    Based on the requirements listed in the src file and the petscconf.h
    info, determine whether this test should be run or not.
    """
    isrun=True
    # MPI requirements
    if testDict.has_key('nsize'):
      if testDict[nsize]>1 and self.conf['MPI_IS_MPIUNI']==1: return False
 
    if testDict.has_key('requires'):
      for requirement in testDict['requires']:
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
        # Precision requirement
        # TODO: int32 is special -- need to figure out
        if requirement in self.precision_types:
          if self.conf['PETSC_PRECISION']==requirement:
            if isNull: return False
          else:
            return False
        # Defines
        if "define(" in requirement:
          pass
        # Rest should be packages


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
