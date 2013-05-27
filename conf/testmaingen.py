#!/usr/bin/env python

import os

PREAMBLE = r"""
#include <stdio.h>
#include <string.h>

int usage(const char *argv0) {
  fprintf(stderr,"Usage: %s path/to/example\n", argv0);
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 2) return usage(argv[0]);
"""

EXCALL = """
  if (!strcmp(argv[1], "%(path)s")) {
    extern int %(mangle)s(int, char**);
    return %(mangle)s(argc-1, &argv[1]);
  }
"""

EPILOGUE = r"""
  fprintf(stderr,"Unknown test or tutorial: %s\n", argv[1]);
  return 2;
}
"""

def mangle(sourcefile):
    def fmangle(main, ext):
        if ext == '.F':         # Need better mangling support
            return main + '_'
        return main
    stem, ext = os.path.splitext(sourcefile)
    main = 'main_' + stem.replace(os.path.sep, '_').replace('-','_') # '/' and '-' are allowed in paths
    # Shorten names to try to stay within 31 characters
    main = main.replace('_src_','_') # Redundant because all the examples are under src
    main = main.replace('_examples_tests_', '_t_')
    main = main.replace('_examples_tutorials_', '_u_')
    return fmangle(main, ext)

def gen_main(f, sources):
    f.write(PREAMBLE)
    for s in sources:
        f.write(EXCALL % dict(path=s, mangle=mangle(s)))
    f.write(EPILOGUE)

def main(output, sources):
    f = open(output, 'w')
    try:
        gen_main(f, sources)
    finally:
        f.close()

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-o', '--output', help='Location to write test main', default=None)
    opts, extra_args = parser.parse_args()
    if not opts.output:
        parser.error('Output not specified')
    main(output=opts.output, sources=extra_args)
