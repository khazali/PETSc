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

class convertExamples(PETScExamples):
  def __init__(self,replaceSource):
    super(convertExamples, self).__init__()
    self.replaceSource=replaceSource
    self.writeScripts=True
    #self.scriptsSubdir=""
    self.scriptsSubdir="from_makefile"
    return

  def fixScript(self,scriptStr,varVal):
    """
    makefile is commands are not proper bash so need to fix that
    Our naming scheme is slightly different as well, so fix that
    Simple replaces done here -- this is not sophisticated
    Also, this may contain variables defined in makefile and need to do a
    substitution
    """
    scriptStr=scriptStr.replace("then","; then")
    #scriptStr=scriptStr.strip().replace("\\","")
    scriptStr=scriptStr.replace("; \\","\n")
    scriptStr=scriptStr.replace("do\\","do\n")
    scriptStr=scriptStr.replace("do \\","do\n")
    scriptStr=scriptStr.replace("done;\\","done\n")
    # Note the comment out -- I like to see the output
    scriptStr=scriptStr.replace("-@","")
    # Thsi is for ts_eimex*.sh
    scriptStr=scriptStr.replace("$$","$")
    if 'for' in scriptStr:
      scriptStr=scriptStr.replace("$(seq","")
      scriptStr=scriptStr.replace(");",";")
    #if '(${DIFF}' in scriptStr:
    #  scriptStr=scriptStr.split("(")[1]
    #  scriptStr=scriptStr.split(")")[0]
    #  tmpscriptStr=sh.readline()
    if '${DIFF}' in scriptStr.lower() and '||' in scriptStr:
      scriptStr=scriptStr.split("||")[0].strip()
    for var in varVal.keys():
      if var in scriptStr:
        replStr="${"+var+"}"
        scriptStr=scriptStr.replace(replStr,varVal[var])
    scriptStr=scriptStr.replace("\n\n","\n")
    return scriptStr

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

  def abstractScript(self,runexName,scriptStr,anlzDict):
    """
    Abstract script tries to take a normal script string and then parse it
    out such that  it is in the new format
    Abstraction is to fill out:
      NSIZE, ARGS, OUTPUT_SUFFIX 
    If we can't abstract, we can't abstract and we'll just put in a script
    """
    abstract={}
    abstract['abstracted']=False
    abstract['script']=scriptStr.strip()

    # OUTPUT SUFFIX should be equivalent to runex target so we just go by that.
    runexBase=runexName.split("_")[0]
    abstract['outputSuffix']=runexName[len(runexBase)+1:]

    # Things that we know off the bat destroy the abstraction
    cleanStr=re.sub('"[^>]+"', '', scriptStr)
    if "for " in cleanStr:  return abstract
    if "grep " in cleanStr:  return abstract
    if "sort " in cleanStr:  return abstract
    if cleanStr.count("MPIEXEC") > 1: return abstract
    if not "2>" in cleanStr: return abstract   # Abstraction requires redirect

    # Now parse out the stuff
    mainCommand=scriptStr.split("2>")[0].strip()
    if not mainCommand: return abstract

    # Args
    firstPart=mainCommand.split(">")[0]
    exName=runexBase[3:]
    args=firstPart.split(exName)[1].strip().strip("\\")
    if args.strip(): 
      if "\\\n" in args:
        abstract['args']=""
        for a in args.split("\\\n"):
          abstract['args']=abstract['args']+" "+a.strip().strip("\\")
      else:
        abstract['args']=args.strip()

    # nsize
    if "MPIEXEC" in firstPart:
      splitVar=exName
      if "touch" in firstPart: splitVar="./"+exName 
      mpiStuff=firstPart.split(splitVar)[0].strip()
      nsizeStr=mpiStuff.split("-n")[1].split()[0].strip()
      try:
        abstract['nsize']=int(nsizeStr)
      except:
        return abstract
    else:
      abstract['nsize']=0

    abstract['abstracted']=True
    return abstract

  def insertScriptIntoSrc(self,runexName,basedir,subDict):
    """
    This is a little bit tricky because figuring out the source file
    associated with a run file is not done by order in TEST* variables.
    Rather than try and parse that logic, we will try to 
    """
    debug=False
    abstract=subDict[runexName]
    startdir=os.path.realpath(os.path.curdir)
    os.chdir(basedir)
    exBase=runexName[3:].split("_")[0]
    if subDict[runexName].has_key('source'):
      exSrc=subDict[runexName]['source']
    else:
      if debug: print "Can't find source file", basedir,runexName
      subDict['nonUsedTests'].append(runexName)
      os.chdir(startdir)
      return True

    # Before writing out the abstracted info
    # do a final update of requirements based on arguments
    import convertExamplesUtils
    argMap=convertExamplesUtils.argMap
    if abstract['abstracted']:
      if abstract.has_key('args'):
        if abstract.has_key('requires'):
          allRqs=abstract['requires']
        else:
          allRqs=[]
        for matchStr in argMap:
          if matchStr in abstract['args']:
            rqList=argMap[matchStr].split("requires:")[1].strip().split()
            allRqs=allRqs+rqList
        abstract['requires']=list(set(allRqs))  # Uniquify the requirements

    # Get the string to insert
    indent="  "
    if abstract['abstracted']:
      insertStr=indent+"output_suffix: "+abstract['outputSuffix']+"\n"
      insertStr=insertStr+indent+"nsize: "+str(abstract['nsize'])+"\n"
      if abstract.has_key('args'):
        insertStr=insertStr+indent+"args: "+abstract['args']+"\n"
      if abstract.has_key('requires'):
        reqStr=", ".join(abstract['requires'])
        insertStr=insertStr+indent+"requires: "+reqStr+"\n"
    else:
      insertStr=indent+"script: "+abstract['script']+"\n"

    # For now we are writing out to a new file
    newExSrc="new_"+exSrc
    if os.path.exists(newExSrc): exSrc=newExSrc

    # Get the file into a string, append, and then close
    sh=open(exSrc,"r"); fileStr=sh.read(); sh.close()
    newFileStr=fileStr+"\n/*TEST\n"+insertStr+"\nTEST*/\n"
    # Write it out
    sh=open(newExSrc,"w"); sh.write(newFileStr); sh.close()

    os.chdir(startdir)
    return True

  def getVarVal(self,varVal,line,fh):
    """
    
    """
    debug=False
    while 1:
      last_pos=fh.tell()
      if line.strip().endswith("\\"):
        line=line.strip().rstrip("\\")+" "+fh.readline().strip()
      else:
        fh.seek(last_pos)  # might be grabbing next var
        break
    if debug: print "       getVarVal> line ", line
    var=line.split("=")[0].strip()
    valList=line.split("=")[1:]
    val=" ".join(valList).strip()
    varVal[var]=val
    return

  def extractRunFile(self,fh,line,mkfile,varVal):
    """
    Given the file handle which points to the location in the file where
    a runex: has just been read in, write it out to the file in the
    directory where mkfile is located
    Store summary of what we did in the part of dataDict relevant
    """
    debug=False
    runexName=line.split(":")[0].strip()
    #print mkfile, runexName
    alphabet=tuple(string.ascii_letters)
    shStr=""
    basedir=os.path.dirname(mkfile)
    shName=os.path.join(basedir,runexName+".sh")
    while 1:
      last_pos=fh.tell()
      #if runexName=="runex138_2": print line
      line=fh.readline()
      # If it has xterm in it, then remove because we 
      # do not want to test for that
      if not line: break
      #if line.startswith(alphabet): 
      if line.startswith(alphabet) or line.startswith("#"): 
        fh.seek(last_pos)  # might be grabbing next script so rewind
        break
      shStr=shStr+" "+line
    if not shStr.strip(): return "",""
    newShStr=self.fixScript(shStr,varVal)
    # 
    if self.writeScripts:
      if self.scriptsSubdir: 
        subdir=os.path.join(basedir,self.scriptsSubdir)
        if not os.path.isdir(subdir): os.mkdir(subdir)
        shName=os.path.join(subdir,runexName+".sh")
      shh=open(shName,"w")
      shh.write(newShStr)
      shh.close()
      os.chmod(shName,0777)
    return shName, newShStr

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
    # Go through and parse the makefiles
    alphabet=tuple(string.ascii_letters)
    fullmake=os.path.join(root,"makefile")
    fh=open(fullmake,"r")

    # Use examplesAnalyze to get the TEST*scripts
    anlzDict={}
    self.examplesAnalyze(root,dirs,files,anlzDict)
    for var in anlzDict[fullmake]:
      if var == "runexFiles": continue
      varScript=os.path.join(root,var+".sh")
      vh=open(varScript,"w")
      for t in anlzDict[fullmake][var]['tsts']:
        scriptName=t.split("-")[1]+".sh"
        vh.write(scriptName+"\n")
      os.chmod(varScript,0777)

    testDict=self.transformAnalyzeMap(anlzDict[fullmake])

    # This is the main thing.  Go through the makefile, and for each
    # run* target: extract, abstract, insert
    dataDict[fullmake]={}
    dataDict[fullmake]['nonUsedTests']=[]
    i=0
    allRunex=[]
    varVal={}
    if debug: print fullmake
    while 1:
      line=fh.readline()
      if not line: break
      if line.startswith(alphabet) and "=" in line: self.getVarVal(varVal,line,fh)
      if line.startswith("run"): 
        runex,shStr=self.extractRunFile(fh,line,fullmake,varVal)
        runexName=os.path.basename(runex).split(".")[0]
        if debug: print "Working on> ", runexName, " in ", root
        abstract=self.abstractScript(runexName,shStr,anlzDict[fullmake])
        dataDict[fullmake][runexName]=abstract
        if testDict.has_key(runexName):
          dataDict[fullmake][runexName].update(testDict[runexName])
        if insertIntoSrc: 
          self.insertScriptIntoSrc(runexName,root,dataDict[fullmake])
    fh.close()

    #print root,files
    return

def main():
    parser = optparse.OptionParser(usage="%prog [options] startdir")
    parser.add_option('-r', '--replace', dest='replaceSource',
                      action="store_false", 
                      help='Replace the source files.  Default is false')
    parser.add_option('-s', '--startdir', dest='startdir',
                      help='Where to start the recursion',
                      default='')
    parser.add_option('-f', '--functioneval', dest='functioneval',
                      help='Function to evaluate while traversing example dirs: printFiles default), examplesConsistencyEval', 
                      default='')
    options, args = parser.parse_args()

    # Process arguments
    startdir=''
    if len(args) > 1:
      parser.print_usage()
      return
    elif len(args) == 1:
      startdir=args[0]
    else:
      if not options.startdir == '':
        startdir=options.startdir
    if not startdir:
      parser.print_usage()
      return

    pEx=convertExamples(options.replaceSource)
    if not options.functioneval=='':
      pEx.walktree(startdir,action=options.functioneval)
    else:
      pEx.walktree(startdir)

if __name__ == "__main__":
        main()
