#!/usr/bin/env python
#!/bin/env python
#
#    Annoying warning when a 'next' commit is upstream
#
import os, sys

def isgitrepo(petscdir):
  return os.path.exists(os.path.join(petscdir,'.git'))

def getbranch(petscdir):
  if not isgitrepo(petscdir):
    return ''
  bsDir     = os.path.join(petscdir,'config','BuildSystem')
  if not os.path.exists(bsDir):
    return ''
  sys.path.insert(0,bsDir)

  import script

  output,error,retcode = script.Script.runShellCommand('git rev-parse --abbrev-ref HEAD',None,petscdir)

  return output.rstrip()

if __name__ == '__main__':
  if 'PETSC_DIR' in os.environ:
    petscdir = os.environ['PETSC_DIR']
  elif os.path.exists(os.path.join('.', 'include', 'petscversion.h')):
    petscdir = '.'
  else:
    sys.exit(0)
  if not isgitrepo(petscdir):
    sys.exit(0)
  file   = os.path.join(petscdir,'.petscnextupstream')
  branch = getbranch(petscdir)
  if os.path.exists(file) and not branch == 'next':
    print("*******************************************************************************************")
    print("WARNING: The version of the PETSc source you are using cannot be pulled into any of the")
    print("         PETSc git integration branches (maint, master, or next).")
    print("         Your current branch " + branch + " is based on next, or next was merged into it.")
    print("*******************************************************************************************")

