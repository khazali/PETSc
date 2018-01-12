#!/usr/bin/env python
import glob
import sys
import os
import optparse
import re
import inspect


thisfile = os.path.abspath(inspect.getfile(inspect.currentframe()))
pdir = os.path.dirname(os.path.dirname(os.path.dirname(thisfile)))
sys.path.insert(0, os.path.join(pdir, 'config'))

"""
  Simple walker for the reviewing source code files
"""


def getLanguage(self, srcfile):
    """
    Based on the source, determine associated language
    """
    srcext = os.path.splitext(srcfile)[-1]
    return srcext[1:]


def check_manpages(sfile):
    """
    Parse the file for man pages info
    See for example:
      src/vec/vec/utils/tagger/interface/tagger.c

    """
    fh = open(sfile, 'r'); sfile_text = fh.read(); fh.close
    splitfile = re.split('/\*@C', sfile_text)
    if len(splitfile) > 1:
        print '    ===> ', sfile 
        comment_text = re.split('@\*/', splitfile[1])

    return


def walktree(top, action, verbose):
    """
    Walk a directory tree, starting from 'top'
    """
    d = {}
    alldatafiles = []
    for root, dirs, files in os.walk(top, topdown=False):
        # Skip examples -- no source code
        if "examples" in root: continue
        if '.dSYM' in root: continue
        if verbose: print(root)

        for sfile in files:
            # Ignore emacs and other temporary files
            if sfile.startswith("."): continue
            if sfile.startswith("#"): continue
            if 'makefile' in sfile:   continue
            if 'benchmarks' in root:   continue
            if sfile.endswith(".h"): continue    # header files don't have docs?

            # Convenience
            fullex = os.path.join(root, sfile)
            if verbose: print('   --> '+fullex)
            eval(action+"('"+fullex+"')")

    return


def main():
    parser = optparse.OptionParser(usage = "%prog [options] startdir")
    parser.add_option('-s', '--startdir', dest = 'startdir',
                      help = 'Where to start the recursion',
                      default = '')
    parser.add_option('-d', '--datafilespath', dest = 'datafilespath',
                      help='Location of datafilespath for action gen_dl_script',
                      default = None)
    parser.add_option('-a', '--action', dest = 'action',
                      help='action to take from traversing examples: print_datafiles, gen_dl_script',
                      default = 'check_manpages')
    parser.add_option('-p', '--petsc_dir', dest = 'petsc_dir',
                      help = 'Location of petsc_dir',
                      default='')
    parser.add_option('--verbose', action='store_true',
                      help='Show mismatches between makefiles and the filesystem', 
                      default=False)
    parser.add_option('--petsc-arch', help='Set PETSC_ARCH different from environment', default=os.environ.get('PETSC_ARCH'))
    options, args = parser.parse_args()

    # Process arguments

    if options.petsc_dir:
        petsc_dir = options.petsc_dir
    else:
        petsc_dir = pdir

    startdir = os.path.join(petsc_dir, 'src')
    if len(args) > 1:
        parser.print_usage()
        return
    elif len(args) == 1:
        startdir = args[0]
    else:
        if not options.startdir == '':
            startdir = options.startdir

    # Do actual work

    action = 'print_datafiles' if not options.action else options.action
    walktree(startdir, action, options.verbose)


if __name__ == "__main__":
        main()
