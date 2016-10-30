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

"""
Class for extracting the shell scripts from the makefile


If I had programmed this better, methods from examplesWalker.py
that are per directory specific (e.g., parseTESTline) would be here.
Bad foresight on my part.
"""

class makeParse(object):
  def __init__(self):

    self.writeScripts=True
    #self.scriptsSubdir=""
    self.scriptsSubdir="from_makefile"
    self.ptNaming=True
    self.insertIntoSrc=True
    return

  def makeRunDict(self,mkDict):
    """
    parseTESTline produces a dictionary of the form:
      dataDict[TESTEXAMPLES_*]['tsts']
      dataDict[TESTEXAMPLES_*]['srcs']
    Put it into a form that is easier for process:
      runDict[srcfile][runtest]={}
    """
    import convertExamplesUtils
    makefileMap=convertExamplesUtils.makefileMap
    def getReqs(typeList,reqtype):
    """ Determine requirements associated with srcs and tests """
      allReqs=[]
      for atype in typeList:
        if makefileMap.has_key(atype):
          if makefileMap[atype].startswith(reqtype):
            rqList=makefileMap[atype].split("requires:")[1].strip().split()
            allRqs=allRqs+rqList
      if len(allRqs)>0: return list(set(allRqs))  # Uniquify the requirements
      return allRqs

    # As I reorder the dictionary, keep track of the types
    rDict={}
    for extype in mkDict:
      if not extype.startswith("TEST"): continue
      for exfile in mkDict[extype]['srcs']:
        srcfile=self.undoNameSpace(exfile)
        if not rDict.has_key(srcfile):
          rDict[srcfile]={}
          rDict[srcfile][test]['types']=[]
        rDict[srcfile]['types']=extype
        testList=self.findTests(exfile,mkDict[extype]['tsts'])
        for test in testList:
          if not rDict[srcfile].has_key(test):
            rDict[srcfile][test]={}
            rDict[srcfile][test]['types']=[]
          rDict[srcfile][test]['types'].append(extype)
          # Transfer information from makefile, especially script
          if mkDict.has_key(test): rDict[srcfile][test].update(mkDict[test])

    # Now that all types are known, determine requirements
    for sfile in rDict:
      rDict[sfile]['requires']=getReqs(rDict[sfile]['types'],"buildrequires")
      for test in rDict:
        rDict[sfile][test]['requires']=getReqs(rDict[sfile][test]['types'],"requires")

    # Determine tests that are not invoked (in any circumstance)
    rDict['not_tested']=[]
    for mkrun in mkDict:
      if not mkrun.startswith("run"): continue
      found=False
      for sfile in rDict:
        for runex in rDict[sfile]:
          if runex==mkrun: found=True
      if not found: rDict['not_tested'].append(mkrun)


    # Now need
    debug=True
    for runex in runDict:
      runDict[runex]=pEx.abstractScript(runex,runDict[runex])


    return rDict

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

  def getVarVal(self,line,fh):
    """
    Process lines of form var = val.  Mostly have to handle
    continuation lines
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
    valList=line.split("=")[1:]
    val=" ".join(valList).strip()
    return val

  def extractRunFile(self,fh,line,mkfile,varVal):
    """
    Given the file handle which points to the location in the file where
    a runex: has just been read in, write it out to the file in the
    directory where mkfile is located
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
    return runexName, newShStr

  def parseRunsFromMkFile(self,fullmake):
    """
     Parse out the runs and related variables.  
     Return two dictionaries, one with makefile info 
      and one containing all of the tests
    """
    fh=open(fullmake,"r")
    root=os.path.dirname(os.path.realpath(fullmake))
    debug=False

    # Go through the makefile, and for each run* target: 
    #     extract, abstract, insert
    mkDict={}
    i=0
    if debug: print fullmake
    alphabet=tuple(string.ascii_letters)
    while 1:
      line=fh.readline()
      if not line: break
      # Scripts might have substitutions so need to store all of the
      # defined variables in the makefile
      # For TESTS* vars, we want info in special dictionary
      if line.startswith(alphabet) and "=" in line: 
        var=line.split("=")[0].strip()
        # This does substitution to get filenames
        if line.startswith("TEST"):
          mkDict[var]=self.parseTESTline(fh,line,root)
        else:
          mkDict[var]=self.getVarVal(line,fh)
      
      # Only keep the run targets in addition to vars
      # Do some transformation of the script string at this stage
      if line.startswith("run"): 
        runex,shStr=self.extractRunFile(fh,line,fullmake,mkDict)
        if debug: 
          print runex
          print " ", shStr
        mkDict[runex]={}
        mkDict[runex]['script']=shStr
    fh.close()

    # mkDict in form related to parsing files.  Need it in 
    # form easier for generating the tests
    runDict=self.makeRunDict(mkDict)

    return runDict

  def insertScriptIntoSrc(self,runexName,fullmake,abstract):
    """
    Put the abstract info in a dictionary into the source file in a yaml
    format
    """
    debug=False
    basedir=os.path.dirname(os.path.realpath(fullmake))
    # Sanity checks -- goal is to pass wtihout any of these showing
    if not abstract['used']:
      if debug: print "Test not used: ", runexName
      return False, "not_used"

    if not abstract.has_key('source'):
      if debug: print "Can't find source file", runexName
      return False, "no_source"

    if not abstract['abstracted']:
      if debug: print "Warning: Problem with ", runexName
      return False, "other"

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
    return True, ""

  def abstractForLoops(self,scriptStr):
    """
    If it has a for loop, then need to modify the script string
    to use the {{ ... }} syntax
    """
    new=""
    forvars=[]
    forvals={}
    for line in scriptStr.split("\n"):
      if "for " in line:
        fv=line.split("for")[1].split("in")[0].strip()
        forvars.append(fv)
        forvals[fv]=line.split("in")[1].split(";")[0].strip()
      elif " done" in line:
        pass
      else:
        for fv in forvars:
          fl=forvals[fv]
          line=re.sub("\$"+fv+" ","{{"+fl+"}} ",line)
        new=new+line+"\n"
    return new

  def abstractMpiTest(self,runexName,scriptStr,abstract):
    """
    If it has a for loop, then need to modify the script string
    to use the {{ ... }} syntax
    """
    # We always want nsize even if not abstracted 
    firstPart=scriptStr.split(">")[0]
    if "MPIEXEC" in firstPart:
      nsizeStr=firstPart.split(" -n ")[1].split()[0].strip()
      try:
        abstract['nsize']=int(nsizeStr)
      except:
        return abstract
    else:
      return abstract

    # Now parse out the stuff
    mainCommand=scriptStr.split("2>")[0].strip()
    if not mainCommand: return abstract

    # Args
    runexBase=runexName.split("_")[0]
    exName=runexBase[3:]
    firstPart=mainCommand.split(">")[0]
    args=firstPart.split(exName)[1].strip().strip("\\")
    if args.strip(): 
      if "\\\n" in args:
        abstract['args']=""
        for a in args.split("\\\n"):
          abstract['args']=abstract['args']+" "+a.strip().strip("\\")
      else:
        abstract['args']=args.strip()

    # Before writing out the abstracted info
    # do a final update of requirements based on arguments
    import convertExamplesUtils
    argMap=convertExamplesUtils.argMap
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

    # Things that we know off the bat destroy the abstraction
    cleanStr=re.sub('"[^>]+"', '', scriptStr)
    if "grep " in cleanStr:  return abstract
    if "sort " in cleanStr:  return abstract
    if cleanStr.count("MPIEXEC") > 1: return abstract
    if not "2>" in cleanStr: return abstract   # Abstraction requires redirect

    abstract['abstracted']=True

    return abstract

  def abstractMultiMpiTest(self,scriptStr,abstract):
    """
    If it has a for loop, then need to modify the script string
    to use the {{ ... }} syntax
    """
    return abstract

  def abstractScript(self,runexName,abstract):
    """
    Do a preliminary pass of abstracting the script
    Abstract script tries to take a normal script string and then parse it
    out such that  it is in the new format
    Abstraction is to fill out:
      NSIZE, ARGS, OUTPUT_SUFFIX 
    If we can't abstract, we can't abstract and we'll just put in a script
    """
    debug=True

    abstract['abstracted']=False
    scriptStr=abstract['script']

    # OUTPUT SUFFIX should be equivalent to runex target so we just go by that.
    runexBase=runexName.split("_")[0]
    abstract['outputSuffix']=runexName[len(runexBase)+1:]

    # Do for loop first because that is local parsing
    if "for " in scriptStr:
      if debug: print "FOR LOOP", runexName
      scriptStr=self.abstractForLoops(scriptStr)

    # Handle subtests if needed
    if scriptStr.count("MPIEXEC")>1:
      if debug: print "MultiMPI", runexName
      abstract=self.abstractMultiMpiTest(scriptStr,abstract)
    else:
      abstract=self.abstractMpiTest(runexName,scriptStr,abstract)

    return abstract

  def undoNameSpace(self,srcfile):
    """
    Undo the nameSpaceing
    """
    if self.ptNaming:
      nameString=srcfile.split("-")[1]
    else:
      nameString=srcfile
    return nameString

  def nameSpace(self,srcfile,srcdir):
    """
    Because the scripts have a non-unique naming, the pretty-printing
    needs to convey the srcdir and srcfile.  There are two ways of doing
    this.
    """
    if self.ptNaming:
      cdir=srcdir.split('src')[1].lstrip("/").rstrip("/")
      prefix=cdir.replace('/examples/','_').replace("/","_")+"-"
      nameString=prefix+srcfile
    else:
      #nameString=srcdir+": "+srcfile
      nameString=srcfile
    return nameString

  def getTestsStr(self,fileStr):
    """
    Given a string that has the /*TESTS testsStr TESTS*/ 
    embedded within it, return testsStr
    """
    if not "/*TESTS" in fileStr: return fileStr,""
    first=fileStr.split("/*TESTS")[0]
    fsplit=fileStr.split("/*TESTS")[1]
    testsStr=fsplit.split("TESTS*/")[0]
    return first,testsStr

  def getSourceFileName(self,petscName,srcdir):
    """
    Given a PETSc name of the form ex1.PETSc or ex2.F.PETSc 
    find the source file name
    Source directory is needed to handle the fortran
    """
    # Bad spelling
    word=petscName
    if word.rstrip(".PETSc")[-1]=='f':
      newword = word.replace('PETSc','F')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        newword = word.replace('PETSc','F90')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        print "I give up on this fortran file: ", srcdir, word
    elif 'f90' in word:
      newword = word.replace('PETSc','F90')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        newword = word.replace('PETSc','F')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        print "I give up on this f90 file: ", srcdir, word
        newword=""
    # For files like  
    elif os.path.splitext(word)[0].endswith('cu'):
      newword = word.replace('PETSc','cu')
    else:
      # This is a c file required for the 
      newword = word.replace('PETS','')
      # This means there is a bug in the makefile.  Move along
      if not os.path.isfile(os.path.join(srcdir,newword)):
        newword = word.replace('PETSc','cxx')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        print "I give up on this: ", srcdir, word
        newword=""
    return newword

  def findTests(self,srcfile,testList):
    """
    Given a source file of the form ex1.c and a list of tests of the form
    ['runex1', 'runex1_1', 'runex10', ...]
    Return the list of tests that should be associated with that srcfile
    """
    mtch=os.path.splitext(srcfile)[0]
    if self.ptNaming: mtch=mtch.split("-")[1]
    newList=[]
    for test in testList:
      if self.ptNaming: test=test.split("-")[1]
      if test.split("_")[0][3:]==mtch: newList.append(test)
    return newList

  def parseTESTline(self,fh,line,srcdir):
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
    parseDict={}; parseDict['srcs']=[]; parseDict['tsts']=[]
    debug=False
    while 1:
      last_pos=fh.tell()
      if line.strip().endswith("\\"):
        line=line.strip().rstrip("\\")+" "+fh.readline().strip()
      else:
        fh.seek(last_pos)  # might be grabbing next var
        break
    if debug: print "       parseTESTline> line ", line
    # Clean up the lines to only have a dot-c name
    justfiles=line.split("=")[1].strip()
    justfiles=justfiles.split("#")[0].strip() # Remove comments
    if len(justfiles.strip())==0: return parseDict
    examplesList=justfiles.split(" ")
    # Now parse the line and put into lists
    srcList=[]; testList=[]; removeList=[]
    for exmpl in examplesList:
      if len(exmpl.strip())==0: continue
      if exmpl.endswith(".PETSc"): 
        srcfile=self.getSourceFileName(exmpl,srcdir)
        parseDict[srcfile]=[] # Create list of tests assocated with src file
        srcList.append(self.nameSpace(srcfile,srcdir))
      elif exmpl.startswith("run"): 
        testList.append(self.nameSpace(exmpl,srcdir))
        parseDict[srcfile].append(exmpl)
      elif exmpl.endswith(".rm"): 
        removeList.append(exmpl) # Can remove later if needed
      else:
        srcList.append(self.nameSpace(exmpl,srcdir))
    if debug: print "       parseTESTline> ", srcList, testList
    #if "pde_constrained" in srcdir: raise ValueError('Testing')
    parseDict['srcs']=srcList
    parseDict['tsts']=testList
    return parseDict
   
def printMkParseDict(mkDict,runDict):
  """
  This is for debugging
  """
  indent="  "
  for key in mkDict: print key,": ", str(mkDict[key])
  print "\n\n"

  for runex in runDict:
    print runex
    for rkey in runDict[runex]:
      print indent+rkey,": ", str(runDict[runex][rkey])
    print "\n"
  return 

def main():
    parser = optparse.OptionParser(usage="%prog [options] startdir")
    parser.add_option('-p', '--petsc_dir', dest='petsc_dir',
                      help='Where to start the recursion',
                      default='')
    parser.add_option('-m', '--makefile', dest='makefile',
                      help='Function to evaluate while traversing example dirs: genAllRunFiles cleanAllRunFiles', 
                      default='makefile')
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

    if not options.makefile: print "Use -m to specify makefile"
    pEx=makeParse()
    runDict=pEx.parseRunsFromMkFile(options.makefile)
    #TMP printMkParseDict(runDict)

    total=0; fail=0
    for runex in runDict:
      total=total+1
      # If requested, then insert into source
      if pEx.insertIntoSrc: 
        stat,err=pEx.insertScriptIntoSrc(runex,options.makefile,runDict[runex])
        if not stat: 
          #print runex, err
          fail=fail+1
    print str(fail)+" out of "+str(total)+" test conversions failed."

if __name__ == "__main__":
        main()
