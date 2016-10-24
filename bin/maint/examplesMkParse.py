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
Class for extracting the shell scripts from the makefile


If I had programmed this better, methods from examplesWalker.py
that are per directory specific (e.g., parseline) would be here.
Bad foresight on my part.
"""

class makeParse(object):
  def __init__(self):
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

  def genRunsFromMakefile(self,fullmake):
    """
     Because the generation of the new source files requires 
    """
    fh=open(fullmake,"r")
    debug=False

    # Go through the makefile, and for each run* target: 
    #     extract, abstract, insert
    mkDict={}
    i=0
    varVal={}
    if debug: print fullmake
    alphabet=tuple(string.ascii_letters)
    while 1:
      line=fh.readline()
      if not line: break
      # Scripts might have substitutions so need to store all of the
      # defined variables in the makefile
      if line.startswith(alphabet) and "=" in line: self.getVarVal(varVal,line,fh)
      if line.startswith("run"): 
        runex,shStr=self.extractRunFile(fh,line,fullmake,varVal)
        if debug: 
          print runex
          print " ", shStr
        mkDict[runex]={}
        mkDict[runex]['script']=shStr
    fh.close()

    return mkDict

  def printMkParseDict(self,mkDict):
    """
    This is for debugging
    """
    indent="  "
    for runex in mkDict:
      print runex
      for rkey in mkDict[runex]:
        print indent+rkey
        print indent*3+str(mkDict[runex][rkey])
      print "\n\n"
    return 

  def abstractScript(self,runexName,scriptStr):
    """
    Do a preliminary pass of abstracting the script
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
    exName=runexBase[3:]
    abstract['outputSuffix']=runexName[len(runexBase)+1:]

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
    args=firstPart.split(exName)[1].strip().strip("\\")
    if args.strip(): 
      if "\\\n" in args:
        abstract['args']=""
        for a in args.split("\\\n"):
          abstract['args']=abstract['args']+" "+a.strip().strip("\\")
      else:
        abstract['args']=args.strip()

    abstract['abstracted']=True

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

    return abstract

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
    mkDict=pEx.genRunsFromMakefile(options.makefile)
    # Now for each runex target, abstract it
    debug=True
    for runex in mkDict:
      mkDict[runex]=pEx.abstractScript(runex,mkDict[runex]['script'])
    pEx.printMkParseDict(mkDict)

if __name__ == "__main__":
        main()
