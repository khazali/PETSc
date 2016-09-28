#!/usr/bin/env python
import glob
import sys
import os
import stat
import types
import optparse
import string

"""
Quick start
-----------

Architecture independent (just analyze makefiles):

  ./examplesWalker.py -f examplesAnalyze src
     Show tests source files and tests as organized by type (TESTEXAMPLES*), 
     Output: 
        MakefileAnalysis-tests.txt      MakefileSummary-tests.txt
        MakefileAnalysis-tutorials.txt  MakefileSummary-tutorials.txt

  ./examplesWalker.py -f examplesConsistency src
     Show consistency between the EXAMPLES* variables and TESTEXAMPLES* variables
     Output: 
        ConsistencyAnalysis-tests.txt      ConsistencySummary-tests.txt
        ConsistencyAnalysis-tutorials.txt  ConsistencySummary-tutorials.txt

Architecture dependent (need to modify files and run tests - see below):

  ./examplesWalker.py -f getFileSizes src
     Give the example file sizes for the tests (see below)
     Output:  FileSizes.txt

  ./examplesWalker.py -m <output of make alltests>
     Show examples that are run
     Output:  
        RunArchAnalysis-tests.txt      RunArchSummary-tests.txt
        RunArchAnalysis-tutorials.txt  RunArchSummary-tutorials.txt

  ./examplesWalker.py -f genRunFiles -m <output of make alltests>
     Generate run files as shell scripts in each test/examples directory
     Output:  
        Shell scripts in each directory

Description
------------------

This is a simple os.walk through the examples with various 
"actions" that can be performed on them.  Each action 
is to fill a dictionary as its output.  The signature of
the action is::

   action(root,dirs,files,dict)

For every action, there is an equivalent <action>_summarize
that then takes the dictionary and either prints or 
writes to a file.  The signature is::

   action_summarize(dict)

Currently the actions are: 
    printFiles
      -- Just a simple routine to demonstrate the walker
    examplesAnalyze
      -- Go through summarize the state of the examples
      -- Does so in an architecture independent way
    examplesConsistency
      -- Go through makefiles and see if the documentation
         which uses EXAMPLES* is consistent with what is tested
         by TESTEXAMPLES*
    getFileSizes
      -- After editing lib/petsc/conf/rules to redefine RM (see below),
         you can keep the executables around.  Then specify this
         action to calculate the file sizes of those executables

One other mode:
  ./examplesWalker.py -m <output of make alltests>
      -- If lib/petsc/conf/tests is modified to turn on output
         the tests run, then one can see the tests are written
         out to the summary.  We then process that output and 
         summarize the results (using examplesAnalyze_summarize)
"""

testtypes=[]
testtypes.append("TESTEXAMPLES_C")
testtypes.append("TESTEXAMPLES_FORTRAN")
testtypes.append("TESTEXAMPLES_F90")
testtypes.append("TESTEXAMPLES_F90_NOCOMPLEX")
testtypes.append("TESTEXAMPLES_F2003")
testtypes.append("TESTEXAMPLES_C_COMPLEX")
testtypes.append("TESTEXAMPLES_FORTRAN_COMPLEX")
testtypes.append("TESTEXAMPLES_C_NOCOMPLEX")
testtypes.append("TESTEXAMPLES_FORTRAN_NOCOMPLEX")
testtypes.append("TESTEXAMPLES_C_X")
testtypes.append("TESTEXAMPLES_FORTRAN_MPIUNI")
testtypes.append("TESTEXAMPLES_C_X_MPIUNI")
testtypes.append("TESTEXAMPLES_C_NOTSINGLE")
# These reuse the executables
specific_tests=['CUDA','CUSP','CUSPARSE','HYPRE','MUMPS','SUPERLU','SUPERLU_DIST','MKL_PARDISO']
testtypes.append("TESTEXAMPLES_CUDA")
testtypes.append("TESTEXAMPLES_CUSP")
testtypes.append("TESTEXAMPLES_CUSPARSE")
testtypes.append("TESTEXAMPLES_HYPRE")
testtypes.append("TESTEXAMPLES_MUMPS")
testtypes.append("TESTEXAMPLES_SUPERLU")
testtypes.append("TESTEXAMPLES_SUPERLU_DIST")
testtypes.append("TESTEXAMPLES_MKL_PARDISO")
#testtypes.append("TESTEXAMPLES_MOAB")
#testtypes.append("TESTEXAMPLES_THREADCOMM")

ptNaming=True

def fixScript(scriptStr):
  """
  makefile is commands are not proper bash so need to fix that
  Our naming scheme is slightly different as well, so fix that
  Simple replaces done here -- this is not sophisticated
  """
  scriptStr=scriptStr.replace("then","; then")
  #scriptStr=scriptStr.strip().replace("\\","")
  # Need to generalize
  scriptStr=scriptStr.replace("${MPIEXEC}","mpiexec") 
  scriptStr=scriptStr.replace("${NP}","1") # See ksp/ksp/makefile
  scriptStr=scriptStr.replace("${MDOMAINS}","2") # See ksp/ksp/makefile
  scriptStr=scriptStr.replace("${NDOMAINS}","2") # See ksp/ksp/makefile
  scriptStr=scriptStr.replace("${BREAKPOINT}","") # See ksp/ksp/makefile
  scriptStr=scriptStr.replace("${DATAFILESPATH}/matrices","data")
  scriptStr=scriptStr.replace("${DIFF}","ndiff")
  scriptStr=scriptStr.replace("${SED}","sed")
  # General makefile fixes
  scriptStr=scriptStr.replace("; \\","\n")
  scriptStr=scriptStr.replace("do\\","do\n")
  scriptStr=scriptStr.replace("do \\","do\n")
  scriptStr=scriptStr.replace("done;\\","done\n")
  # Note the comment out -- I like to see the output
  scriptStr=scriptStr.replace("${RM}","#/bin/rm")
  scriptStr=scriptStr.replace("-@","")
  # Thsi is for ts_eimex*.sh
  scriptStr=scriptStr.replace("$$","$")
  if 'for' in scriptStr:
    scriptStr=scriptStr.replace("$(seq","")
    scriptStr=scriptStr.replace(");",";")
  # Do plain diffs to let CTest capture the error message
  if '(diff' in scriptStr:
    scriptStr=scriptStr.split("(")[1]
    scriptStr=scriptStr.split(")")[0]
    tmpscriptStr=sh.readscriptStr()
  if 'diff' in scriptStr and '||' in scriptStr:
    scriptStr=scriptStr.split("||")[0].strip()
  return scriptStr

def insertScriptIntoSrc(runexName,newShStr,basedir):
  """
  This is a little bit tricky because figuring out the source file
  associated with a run file is not done by order in TEST* variables.
  Rather than try and parse that logic, we will try to 
  """
  startdir=os.path.realpath(os.path.curdir)
  os.chdir(basedir)
  exBase=runexName[3:].split("_")[0]
  if exBase.endswith("f") or exBase.endswith("f90"):
    found=False
    for ext in ["F","F90"]:
      exSrc=exBase+"."+ext
      if os.path.exists(exSrc): 
        found=True
        break
  else:
    found=False
    for ext in ["c","cxx","cuda"]:
      exSrc=exBase+"."+ext
      if os.path.exists(exSrc): 
        found=True
        break
  if not found: 
    print "Cannot find source file for ", runexName, " in ", basedir
    os.chdir(startdir)
    return

  # For now we are writing out to a new file
  newExSrc="new_"+exSrc
  if os.path.exists(newExSrc): exSrc=newExSrc

  # Get the file into a string
  sh=open(exSrc,"r"); fileStr=sh.read(); sh.close()

  # Insert the run file
  firstPart=fileStr.split("#include")[0]
  secndPart=fileStr[len(firstPart):]
  newFileStr=firstPart+"\n/*TEST\n"+newShStr+"\nTEST*/\n"+secndPart

  # Write it out
  sh=open(newExSrc,"w"); sh.write(newFileStr); sh.close()

  os.chdir(startdir)
  return 

def extractRunFile(fh,line,mkfile):
  """
  Given the file handle which points to the location in the file where
  a runex: has just been read in, write it out to the file in the
  directory where mkfile is located
  """
  insertIntoSrc=True
  runexName=line.split(":")[0].strip()
  #print mkfile, runexName
  alphabet=tuple(string.ascii_letters)
  shStr=""
  basedir=os.path.dirname(mkfile)
  shName=os.path.join(basedir,runexName+".sh")
  while 1:
    last_pos=fh.tell()
    line=fh.readline()
    # If it has xterm in it, then remove because we 
    # do not want to test for that
    if not line: break
    if line.startswith(alphabet): 
      fh.seek(last_pos)  # might be grabbing next script so rewind
      break
    shStr=shStr+" "+line
  newShStr=fixScript(shStr)
  shh=open(shName,"w")
  shh.write(newShStr)
  shh.close()
  if insertIntoSrc: insertScriptIntoSrc(runexName,newShStr,basedir)
  os.chmod(shName,0777)
  return shName

def nameSpace(srcfile,srcdir):
  """
  Because the scripts have a non-unique naming, the pretty-printing
  needs to convey the srcdir and srcfile.  There are two ways of doing
  this.
  """
  if ptNaming:
    cdir=srcdir.split('src')[1].lstrip("/").rstrip("/")
    prefix=cdir.replace('/examples/','_').replace("/","_")+"-"
    nameString=prefix+srcfile
  else:
    #nameString=srcdir+": "+srcfile
    nameString=srcfile
  return nameString

def getSourceFileName(petscName,srcdir):
  """
  Given a PETSc name of the form ex1.PETSc or ex2.F.PETSc 
  find the source file name
  Source directory is needed to handle the fortran
  """
  word=petscName
  if word.rstrip(".PETSc")[-1]=='f':
    newword = word.replace('PETSc','F')
  elif 'f90' in word:
    newword = word.replace('PETSc','F90')
    if not os.path.isfile(os.path.join(srcdir,newword)):
      newword = word.replace('PETSc','F')
    if not os.path.isfile(os.path.join(srcdir,newword)):
      print "I give up on this: ", srcdir, word
      newword=""
  # For files like  
  elif os.path.splitext(word)[0].endswith('cu'):
    newword = word.replace('PETSc','cu')
  else:
    # This is a c file required for the 
    newword = word.replace('PETS','')
    # This means there is a bug in the makefile.  Move along
    if not os.path.isfile(os.path.join(srcdir,newword)):
      print "I give up on this: ", srcdir, word
      newword=""
  return newword

def findTests(srcfile,testList):
  """
  Given a source file of the form ex1.c and a list of tests of the form
  ['runex1', 'runex1_1', 'runex10', ...]
  Return the list of tests that should be associated with that srcfile
  """
  mtch=os.path.splitext(srcfile)[0]
  if ptNaming: mtch=mtch.split("-")[1]
  newList=[]
  for test in testList:
    if ptNaming: test=test.split("-")[1]
    if test.split("_")[0][3:]==mtch: newList.append(test)
  return newList

def parseline(fh,line,srcdir):
  """
  For a line of the form:
    VAR = ex1.PETSc runex1
  return two lists of the source files and run files
  getSourceFileName is used to change PETSc into the 
  appropriate file extension
   - fh is the file handle to the makefile
   - srcdir is where the makefile and examples are located
  Note for EXAMPLESC and related vars, it ex1.c instead of ex1.PETSc
  """
  debug=False
  while 1:
    if line.strip().endswith("\\"):
      line=line.strip().rstrip("\\")+" "+fh.readline().strip()
    else:
      break
  if debug: print "       parseline> line ", line
  # Clean up the lines to only have a dot-c name
  justfiles=line.split("=")[1].strip()
  justfiles=justfiles.split("#")[0].strip() # Remove comments
  if len(justfiles.strip())==0: return [],[]
  examplesList=justfiles.split(" ")
  # Now parse the line and put into lists
  srcList=[]; testList=[]; removeList=[]
  for exmpl in examplesList:
    if len(exmpl.strip())==0: continue
    if exmpl.endswith(".PETSc"): 
      srcfile=getSourceFileName(exmpl,srcdir)
      srcList.append(nameSpace(srcfile,srcdir))
    elif exmpl.startswith("run"): 
      testList.append(nameSpace(exmpl,srcdir))
    elif exmpl.endswith(".rm"): 
      removeList.append(exmpl) # Can remove later if needed
    else:
      srcList.append(nameSpace(exmpl,srcdir))
  if debug: print "       parseline> ", srcList, testList
  #if "pde_constrained" in srcdir: raise ValueError('Testing')
  return srcList, testList
 
 
def printFiles_summarize(dataDict):
  """
   Simple example of an action
  """
  for root in dataDict:
    print root+": "+" ".join(dataDict[root])
  return

def printFiles(root,dirs,files,dataDict):
  """
   Simple example of an action
  """
  dataDict[root]=files
  return

def getFileSizes_summarize(dataDict):
  """
   Summarize the file sizes
  """
  fh=open("FileSizes.txt","w")
  totalSize=0
  nfiles=0
  toMBorKB=1./1024.0
  for root in dataDict:
    for f in dataDict[root]:
      size=dataDict[root][f]*toMBorKB
      fh.write(f+": "+ "%.1f" % size +" KB\n")
      totalSize=totalSize+size
      nfiles=nfiles+1
  totalSizeMB=totalSize*toMBorKB
  fh.write("----------------------------------------\n")
  fh.write("totalSize = "+ "%.1f" % size +" MB\n")
  fh.write("Number of execuables = "+str(nfiles)+"\n")
  return

def getFileSizes(root,dirs,files,dataDict):
  """
   If you edit this file:
     lib/petsc/conf/rules
    and add at the bottom:
     RM=echo
   Then you will leave the executables in place.
   Once they are in place, run this script and you will get a summary of
   the file sizes
  """
  # Find executables
  xFiles={}
  for fname in files:
    f=os.path.join(root,fname)
    if os.access(f,os.X_OK):
      xFiles[f]=os.path.getsize(f)
  if len(xFiles.keys())>0: dataDict[root]=xFiles
  return

def getCorrespondingKeys(extype):
  """
  Which TESTEX* vars should we look for given an EXAMPLE* variable
  """
  if extype=='EXAMPLESC':
    return ['TESTEXAMPLES_C','TESTEXAMPLES_C_X']
  elif extype=='EXAMPLESF':
    return ['TESTEXAMPLES_F','TESTEXAMPLES_FORTRAN']
  else:
    raise "Error: Do not know extype "+extype
  return

def getMatchingKeys(extype,mkDict):
  """
   Given EXAMPLES*, see what matching keys are in the dict
  """
  mtchList=[]
  for ckey in getCorrespondingKeys(extype):
    if mkDict.has_key(ckey): mtchList.append(ckey)
  return mtchList

def examplesConsistency_summarize(dataDict):
  """
   Go through makefile and see where examples 
  """
  indent="  "
  for type in ["tutorials","tests"]:
    fh=open("ConsistencyAnalysis-"+type+".txt","w")
    gh=open("ConsistencySummary-"+type+".txt","w")
    nallsrcs=0; nalltsts=0
    for mkfile in dataDict:
      if not type in mkfile: continue
      fh.write(mkfile+"\n")
      gh.write(mkfile+"\n")
      for extype in ['EXAMPLESC','EXAMPLESF']:
        matchKeys=getMatchingKeys(extype,dataDict[mkfile])
        # Check to see if this mkfile even has types
        if not dataDict[mkfile].has_key(extype):
          if len(matchKeys)>0:
            foundKeys=" ".join(matchKeys)
            fh.write(indent*2+foundKeys+" found BUT "+extype+"not found\n")
            gh.write(indent*2+extype+"should be documented here\n")
          else:
            continue # Moving right along
        # We have the key
        else:
          if len(matchKeys)==0:
            fh.write(indent*2+extype+" found BUT no corresponding types found\n")
            gh.write(indent*2+extype+" is documented without testing under any PETSC_ARCH\n")
            continue # Moving right along
          matchList=[]; allTests=[]
          for mkey in matchKeys:
            matchList=matchList+dataDict[mkfile][mkey]['srcs']
            allTests=allTests  +dataDict[mkfile][mkey]['tsts']
          fh.write(indent+extype+"\n")
          nsrcs=0
          for exfile in dataDict[mkfile][extype]['srcs']:
             if exfile in matchList: 
               matchList.remove(exfile)
               testList=findTests(exfile,allTests)
               ntests=len(testList)
               if ntests==0:
                 fh.write(indent*2+exfile+" found BUT no tests found\n")
               else:
                 tests=" ".join(testList)
                 fh.write(indent*2+exfile+" found with these tests: "+tests+"\n")
             else:
               nsrcs=nsrcs+1
               fh.write(indent*2+"NOT found in TEST*: "+exfile+"\n")
          lstr=" files are documented without testing under any PETSC_ARCH\n"
          if nsrcs>0: gh.write(indent*2+extype+": "+str(nsrcs)+lstr)
          nsrcs=0
          for mtch in matchList:
            fh.write(indent*2+"In TEST* but not EXAMPLE*: "+mtch+"\n")
            nsrcs=nsrcs+1
          lstr=" files have undocumented tests\n"
          if nsrcs>0: gh.write(indent*2+extype+": "+str(nsrcs)+lstr)
          fh.write("\n")
      fh.write("\n"); gh.write("\n")
    fh.close()
    gh.close()
  #print dataDict
  return

def examplesConsistency(root,dirs,files,dataDict):
  """
   Documentation for examples is generated by what is in
   EXAMPLESC and EXAMPLESF variables (see lib/petsc/conf/rules)
   This goes through and compares what is in those variables
   with what is in corresponding TESTEXAMPLES_* variables
  """
  debug=False
  # Go through and parse the makefiles
  fullmake=os.path.join(root,"makefile")
  fh=open(fullmake,"r")
  dataDict[fullmake]={}
  i=0
  searchTypes=[]
  for stype in ['EXAMPLESC','EXAMPLESF']:
    searchTypes.append(stype)
    searchTypes=searchTypes+getCorrespondingKeys(stype)
  if debug: print fullmake
  while 1:
    line=fh.readline()
    if not line: break
    if not "=" in line:  continue  # Just looking at variables
    var=line.split("=")[0].strip()
    if " " in var: continue        # eliminate bash commands that appear as variables
    if debug: print "  "+var
    if var in searchTypes:
      sfiles,tests=parseline(fh,line,root)
      if debug: print "  "+var, sfiles
      if len(sfiles)>0: 
        dataDict[fullmake][var]={}
        dataDict[fullmake][var]['srcs']=sfiles
        dataDict[fullmake][var]['tsts']=tests
      continue
  fh.close()
  #print root,files
  return

def examplesAnalyze_summarize(dataDict):
  """
   Write out files that are from the result of either the makefiles
   analysis (default with walker) or from run output.
   The run output has a special dictionary key to differentiate the
   output
  """
  indent="  "
  if dataDict.has_key("outputname"):
    baseName=dataDict["outputname"]
  else:
    baseName="Makefile"
  for type in ["tutorials","tests"]:
    fh=open(baseName+"Analysis-"+type+".txt","w")
    gh=open(baseName+"Summary-"+type+".txt","w")
    nallsrcs=0; nalltsts=0
    for mkfile in dataDict:
      if not type in mkfile: continue
      nsrcs=0; ntsts=0
      fh.write(mkfile+"\n")
      runexFiles=dataDict[mkfile]['runexFiles']
      for extype in dataDict[mkfile]:
        if extype == "runexFiles": continue
        fh.write(indent+extype+"\n")
        allTests=dataDict[mkfile][extype]['tsts']
        for exfile in dataDict[mkfile][extype]['srcs']:
           nsrcs=nsrcs+1
           testList=findTests(exfile,allTests)
           ntests=len(testList)
           ntsts=ntsts+ntests
           if ntests==0:
             fh.write(indent*2+exfile+": No tests found\n")
           else:
             tests=" ".join(testList)
             fh.write(indent*2+exfile+": "+tests+"\n")
             for t in testList: 
               if t in runexFiles: runexFiles.remove(t)
        fh.write("\n")
      nrunex=len(runexFiles)
      if nrunex>0:
       runexStr=" ".join(runexFiles)
       fh.write(indent+"RUNEX SCRIPTS NOT USED: "+runexStr+"\n")
      fh.write("\n")
      sumStr=str(nsrcs)+" srcfiles; "+str(ntsts)+" tests"
      if nrunex>0: sumStr=sumStr+"; "+str(nrunex)+" tests not used"
      gh.write(mkfile+": "+sumStr+"\n")
      nallsrcs=nallsrcs+nsrcs; nalltsts=nalltsts+ntsts
    fh.close()
    gh.write("-----------------------------------\n")
    gh.write("Total number of sources: "+str(nallsrcs)+"\n")
    gh.write("Total number of tests:   "+str(nalltsts)+"\n")
    gh.close()
  #print dataDict
  return

def examplesAnalyze(root,dirs,files,dataDict):
  """
   Go through makefile and see what examples and tests are listed
   Dictionary structure is of the form:
     dataDict[makefile]['srcs']=sourcesList
     dataDict[makefile]['tsts']=testsList
  """
  debug=False
  # Go through and parse the makefiles
  fullmake=os.path.join(root,"makefile")
  fh=open(fullmake,"r")
  dataDict[fullmake]={}
  i=0
  allRunex=[]
  if debug: print fullmake
  while 1:
    line=fh.readline()
    if not line: break
    if line.startswith("runex"): allRunex.append(line.split(":")[0])
    if not "=" in line:  continue  # Just looking at variables
    var=line.split("=")[0].strip()
    if " " in var: continue        # eliminate bash commands that appear as variables
    if var in testtypes:
      if debug: print "  >", var
      sfiles,tests=parseline(fh,line,root)
      if len(sfiles)>0: 
        dataDict[fullmake][var]={}
        dataDict[fullmake][var]['srcs']=sfiles
        dataDict[fullmake][var]['tsts']=tests
      continue
  dataDict[fullmake]['runexFiles']=allRunex
  fh.close()
  #if "pde_constrained" in root: raise ValueError('Testing')
  #print root,files
  return

def cleanAllRunFiles_summarize(dataDict):
  """
  Required routine
  """
  return

def cleanAllRunFiles(root,dirs,files,dataDict):
  """
  Cleanup from genAllRunFiles
  """
  for runfile in glob.glob(root+"/run*"): os.remove(runfile)
  for newfile in glob.glob(root+"/new_*"): os.remove(newfile)
  for tstfile in glob.glob(root+"/TEST*.sh"): os.remove(tstfile)
  return

def genAllRunFiles_summarize(dataDict):
  """
   For all of the TESTEXAMPLES* find
  """
  fhname="GenAllRunFiles_summarize.txt"
  fh=open(fhname,"w")
  print "See ", fhname
  for mkfile in dataDict:
    for runex in dataDict[mkfile]['runexFiles']:
      fh.write("  "+ runex+"\n")
    fh.write("\n")
  return

def genAllRunFiles(root,dirs,files,dataDict):
  """
   For all of the TESTEXAMPLES* find
  """
  debug=False
  # Go through and parse the makefiles
  fullmake=os.path.join(root,"makefile")
  fh=open(fullmake,"r")
  dataDict[fullmake]={}
  i=0
  allRunex=[]
  if debug: print fullmake
  while 1:
    line=fh.readline()
    if not line: break
    if line.startswith("run"): 
      runex=extractRunFile(fh,line,fullmake)
      allRunex.append(runex)
  dataDict[fullmake]['runexFiles']=allRunex
  fh.close()
  # Use examplesAnalyze to get the TEST*scripts
  anlzDict={}
  examplesAnalyze(root,dirs,files,anlzDict)
  for var in anlzDict[fullmake]:
    if var == "runexFiles": continue
    varScript=os.path.join(root,var+".sh")
    vh=open(varScript,"w")
    for t in anlzDict[fullmake][var]['tsts']:
      scriptName=t.split("-")[1]+".sh"
      vh.write(scriptName+"\n")
    os.chmod(varScript,0777)

  #print root,files
  return

def walktree(top,action="printFiles"):
  """
  Walk a directory tree, starting from 'top'
  """
  print "action", action
  # Goal of action is to fill this dictionary
  dataDict={}
  for root, dirs, files in os.walk(top, topdown=False):
    if not "examples" in root: continue
    if root.endswith("tests") or root.endswith("tutorials"):
      eval(action+"(root,dirs,files,dataDict)")
    if type(top) != types.StringType:
        raise TypeError("top must be a string")
  # Now summarize this dictionary
  eval(action+"_summarize(dataDict)")
  return

def archTestAnalyze(makeoutput):
  """
  To use:
    In file: lib/petsc/conf/test

    Change this line:
      ALLTESTS_PRINT_PROGRESS = no
    to
      ALLTESTS_PRINT_PROGRESS = debugtest

    And run:: 
      make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH alltests

    Save the output, and run this script on it (-m output).
    Output will be in 
  """
  fh=open(makeoutput)
  # Dictionary datastructure of same form used in examplesAnalyze
  testDict={}
  while 1:
    line=fh.readline()
    if not line: break
    if "Testing:" in line:
      var=line.split("Testing: ")[1].split()[0].strip().upper()
    if "Running examples in" in line:
      exdir=line.split("Running examples in")[1].strip().rstrip(":")
      # This is the full path which is not import so simplify this:
      parentdir=os.path.basename(exdir.split("/src/")[0])
      exdir=exdir.split(parentdir+"/")[1]
      # foo is because parseline expects VAR=stuff pattern
      line="FOO = "+fh.readline()
      srcList,testList=parseline(fh,line,exdir)
      if not testDict.has_key(exdir): 
        testDict[exdir]={}
      testDict[exdir][var]={}
      testDict[exdir][var]['srcs']=srcList
      testDict[exdir][var]['tsts']=testList
  # Now that we have our dictionary loaded up, pretty print it
  testDict["outputname"]="RunArch"
  examplesAnalyze_summarize(testDict)
  return

def main():
    parser = optparse.OptionParser(usage="%prog [options] startdir")
    parser.add_option('-s', '--startdir', dest='startdir',
                      help='Where to start the recursion',
                      default='')
    parser.add_option('-f', '--functioneval', dest='functioneval',
                      help='Function to evaluate while traversing example dirs: printFiles default), examplesConsistencyEval', 
                      default='')
    parser.add_option('-m', '--makeoutput', dest='makeoutput',
                      help='Name of make alttests output file',
                      default='')
    options, args = parser.parse_args()

    # Process arguments
    # The makeoutput option is not a walker so just get it over with
    if not options.makeoutput=='':
      archTestAnalyze(options.makeoutput)
      return
    # Do the walker processing
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
    if not options.functioneval=='':
      walktree(startdir,action=options.functioneval)
    else:
      walktree(startdir)

if __name__ == "__main__":
        main()
