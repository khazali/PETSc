#!/usr/bin/env python
import glob
import sys
import re
import os
import stat
import types
import optparse
import string
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir) 
from examplesWalker import *
from examplesMkParse import *

"""
Tools for converting PETSc examples to new format
Based on examplesWalker.py

Quick start
-----------

  bin/maint/convertExamples.py -f genAllRunFiles src
     - Generate scripts from the makefiles
     - Try to abstract the scripts and put the metadata into the source code

  bin/maint/convertExamples.py -f cleanAllRunFiles src

"""

class convertExamples(PETScExamples,makeParse):
  def __init__(self,replaceSource):
    super(convertExamples, self).__init__()
    self.replaceSource=replaceSource
    self.writeScripts=True
    #self.scriptsSubdir=""
    self.scriptsSubdir="from_makefile"
    return

  def transformAnalyzeMap(self,anlzDict):
    """
    examplesAnalyze dataDict produces a dictionary of the form:
      dataDict[fullmake][TESTEXAMPLES_*]['tsts']
      dataDict[fullmake][TESTEXAMPLES_*]['srcs']
    Put it into a form that is easier for later coding
    """
    newDict={}

    # Insert the source file associated with the test
    #  A little bit worried here as if there is a run* test in two TEST*
    #  targets then I won't get it.
    for extype in anlzDict:
      if extype == "runexFiles": continue
      for exfile in anlzDict[extype]['srcs']:
        testList=self.findTests(exfile,anlzDict[extype]['tsts'])
        for test in testList:
          if not newDict.has_key(test):
            newDict[test]={}
            newDict[test]['types']=[]
          newDict[test]['source']=self.undoNameSpace(exfile)
          newDict[test]['types'].append(extype)

    import convertExamplesUtils
    makefileMap=convertExamplesUtils.makefileMap
    # Clean up the requirements
    for test in newDict:
      allRqs=[]
      for atype in newDict[test]['types']:
        if makefileMap.has_key(atype):
          rqList=makefileMap[atype].split("requires:")[1].strip().split()
          allRqs=allRqs+rqList
      if len(allRqs)>0:
        newDict[test]['requires']=list(set(allRqs))  # Uniquify the requirements
    return newDict

  def insertScriptIntoSrc(self,runexName,basedir,abstract):
    """
    This is a little bit tricky because figuring out the source file
    associated with a run file is done by order in TEST* variables.
    Rather than try and parse that logic, we will try to find it
    """
    # Sanity checks -- goal is to pass wtihout any of these showing
    if not abstract.has_key('source'):
      print "Can't find source file", basedir,runexName
      return False

    if not abstract['abstracted']:
      print "Warning: Problem with ", runexName, " in ", basedir
      return False

    debug=False
    startdir=os.path.realpath(os.path.curdir)
    os.chdir(basedir)
    exBase=runexName[3:].split("_")[0]
    exSrc=abstract['source']

    # Get the string to insert
    indent="  "
    insertStr=indent+"test:\n"
    indent=indent*2
    if abstract['abstracted']:
      if abstract.has_key('outputSuffix'):
        if abstract['outputSuffix']:
          suffix=abstract['outputSuffix']
          insertStr=insertStr+indent+"suffix: "+suffix+"\n"
      if abstract.has_key('outputSuffix'):
        if abstract['nsize']>1:
          insertStr=insertStr+indent+"nsize: "+str(abstract['nsize'])+"\n"
      if abstract.has_key('args'):
        insertStr=insertStr+indent+"args: "+abstract['args']+"\n"
      if abstract.has_key('requires'):
        reqStr=", ".join(abstract['requires'])
        insertStr=insertStr+indent+"requires: "+reqStr+"\n"

    if os.path.splitext(exSrc)[1].lstrip(".").startswith("F"):
      insertStr=insertStr.replace("\n","\n!").rstrip("!")

    # Get the file into a string, append, and then close
    newExSrc="new_"+exSrc
    openFile=(newExSrc if os.path.isfile(newExSrc) else exSrc)
    sh=open(openFile,"r"); fileStr=sh.read(); sh.close()

    # Get current tests within the file
    firstPart,currentTestsStr=self.getTestsStr(fileStr)

    # What gets inserted
    testStr="\n/*TESTS\n"+currentTestsStr.lstrip("\n")+insertStr+"\nTESTS*/\n"

    # Append to the file
    sh=open(newExSrc,"w"); sh.write(firstPart+testStr); sh.close()

    os.chdir(startdir)
    return True

  def cleanAllRunFiles_summarize(self,dataDict):
    """
    Required routine
    """
    return

  def cleanAllRunFiles(self,root,dirs,files,dataDict):
    """
    Cleanup from genAllRunFiles
    """
    if self.writeScripts:
      globstr=root+"/new_*"
      if self.scriptsSubdir:  globstr=root+"/"+self.scriptsSubdir+"/run*"
      for runfile in glob.glob(globstr): os.remove(runfile)
    for newfile in glob.glob(root+"/new_*"): os.remove(newfile)
    for tstfile in glob.glob(root+"/TEST*.sh"): os.remove(tstfile)
    for newfile in glob.glob(root+"/*.tmp*"): os.remove(newfile)
    for newfile in glob.glob(root+"/*/*.tmp*"): os.remove(newfile)
    for newfile in glob.glob(root+"/*/*/*.tmp*"): os.remove(newfile)
    for newfile in glob.glob(root+"/trashz"): os.remove(newfile)
    for newfile in glob.glob(root+"/*/trashz"): os.remove(newfile)
    for newfile in glob.glob(root+"/*/*/trashz"): os.remove(newfile)
    for newfile in glob.glob(root+"/*.mod"): os.remove(newfile)
    return

  def getOrderedKeys(self,subDict):
    """
    It looks nicer to have the keys in an ordered way, but we want
    to make sure we get all of the keys, so do list manipulation here
    """
    firstList=["sourceFile","outputSuffix","abstracted","nsize","args"]
    lastList=["script"]
    keyList=subDict.keys()
    for key in subDict.keys():
      if key in firstList+lastList: keyList.remove(key)
    return firstList+keyList+lastList

  def genAllRunFiles_summarize(self,dataDict):
    """
    Summarize the results.
    """
    indent="  "
    fhname="GenAllRunFiles_summarize.txt"
    fh=open(fhname,"w")
    print "See ", fhname
    for mkfile in dataDict:
      fh.write(mkfile+"\n")
      for runex in dataDict[mkfile]:
        if runex=='nonUsedTests': continue
        fh.write(indent+runex+"\n")
        for key in self.getOrderedKeys(dataDict[mkfile][runex]):
          if not dataDict[mkfile][runex].has_key(key): continue
          s=dataDict[mkfile][runex][key]
          if isinstance(s, basestring):
            line=indent*2+key+": "+s
          elif key=='nsize' or key=='abstracted':
            line=indent*2+key+": "+str(s)
          else:
            line=indent*2+key
          fh.write(line+"\n")
        fh.write("\n")
      line=" ".join(dataDict[mkfile]['nonUsedTests'])
      if len(line)>0:
        fh.write(indent+"Could not insert into source from "+mkfile+": "+line+"\n")
      fh.write("\n")
    return

  def genAllRunFiles(self,root,dirs,files,dataDict):
    """
     For all of the TESTEXAMPLES* find the run* targets, convert to
     script, abstract if possible, and create new_ex* source files that
     have the abstracted info.  

     Because the generation of the new source files requires 
    """
    # Because of coding, clean up the directory before parsing makefile
    self.cleanAllRunFiles(root,dirs,files,{})

    debug=False
    insertIntoSrc=True

    # Information comes form makefile
    fullmake=os.path.join(root,"makefile")

    # Use examplesAnalyze to get info from TEST* targets, 
    # but then put it into more useful data structure
    # Provides: associated sourcefile and requirements based on TEST* variables
    anlzDict={}
    self.examplesAnalyze(root,dirs,files,anlzDict)
    testDict=self.transformAnalyzeMap(anlzDict[fullmake])

    # Go through the makefile, and for each run* target: 
    #     extract, abstract, insert
    dataDict[fullmake]={}
    dataDict[fullmake]['nonUsedTests']=[]
    i=0
    varVal={}
    if debug: print fullmake
    # This gets all of the run* targets in makefile.  
    # Can be tested independently in examplesMkParse.py
    mkDict=self.genRunsFromMakefile(fullmake)

    # Now for each runex target, abstract and insert
    for runex in mkDict:
      # Preliminary abstract
      dataDict[fullmake][runex]=self.abstractScript(runex,mkDict[runex]['script'])

      # Update the abstract info from testDict above (source, requires)
      if testDict.has_key(runex):
        dataDict[fullmake][runex].update(testDict[runex])

      # If requested, then insert into source
      if insertIntoSrc: 
        if not self.insertScriptIntoSrc(runex,root,dataDict[fullmake][runex]):
           dataDict[fullmake]['nonUsedTests'].append(runex)

    return

def main():
    parser = optparse.OptionParser(usage="%prog [options] startdir")
    parser.add_option('-r', '--replace', dest='replaceSource',
                      action="store_false", 
                      help='Replace the source files.  Default is false')
    parser.add_option('-p', '--petsc_dir', dest='petsc_dir',
                      help='Where to start the recursion',
                      default='')
    parser.add_option('-f', '--functioneval', dest='functioneval',
                      help='Function to evaluate while traversing example dirs: genAllRunFiles cleanAllRunFiles', 
                      default='genAllRunFiles')
    options, args = parser.parse_args()

    # Process arguments
    if len(args) > 0:
      parser.print_usage()
      return

    petsc_dir=None
    if options.petsc_dir: petsc_dir=options.petsc_dir
    if petsc_dir is None: petsc_dir=os.path.dirname(os.path.dirname(currentdir))
    # This is more inline with what PETSc devs use, but since we are
    # experimental, I worry about picking up their env var
#    if petsc_dir is None:
#      petsc_dir = os.environ.get('PETSC_DIR')
#      if petsc_dir is None:
#        petsc_dir=os.path.dirname(os.path.dirname(currentdir))

    startdir=os.path.join(petsc_dir,'src')
    pEx=convertExamples(options.replaceSource)
    if not options.functioneval=='':
      pEx.walktree(startdir,action=options.functioneval)
    else:
      pEx.walktree(startdir)

if __name__ == "__main__":
        main()
