#!/usr/bin/env python2

# This file generates tests that output TAP format, which can be run and
# tabulated using the 'prove' utility from the Perl distribution.
#
# http://podwiki.hexten.net/TAP/TAP13.html?page=TAP13

from __future__ import print_function

import sys
sys.path.append('/home/jed/src/pymake')
import os
import collections
import pymake.parser
import distutils.sysconfig
import itertools

Link = collections.namedtuple('Link', 'lang out objs libs')
Example = collections.namedtuple('Example', 'out srcs libs')
Run = collections.namedtuple('Run', 'name exe np args cmpname')

def printverbose(*args):
    print(*args)

def parselinkargs(token, isfunc):
    assert not isfunc
    args = token.split()
    assert args[0] == '-o'
    out = args[1]
    objs = args[2:]
    return out, objs

def parselib(token, isfunc):
    assert isfunc and isinstance(token, pymake.functions.VariableRef)
    return token.vname.s

def parselink(command):
    cmd = iter(command)
    token, isfunc = next(cmd)
    assert token == '-'

    token, isfunc = next(cmd)
    assert isfunc and isinstance(token, pymake.functions.VariableRef)
    linker = token.vname.s
    lang = dict(CLINKER='C', FLINKER='F')[linker]

    out, objs = parselinkargs(*next(cmd))
    libs = []
    for token, isfunc in cmd:
        lib = parselib(token, isfunc)
        libs.append(lib)
    return Link(lang, out, objs, libs)

def parserm(command):
    token, isfunc = command[0]
    if isfunc and isinstance(token, pymake.functions.VariableRef) and token.vname.s == 'RM':
        return True
    else:
        import pdb; pdb.set_trace()
        return False

def findsrc(obj):
    base, oext = os.path.splitext(obj)
    assert oext == '.o'
    extmap = [('f', 'F F90'.split()), ('f90', 'F90 F95 F'.split()), ('', 'c cxx'.split())]
    for k, exts in extmap:
        if not base.endswith(k): continue
        for ext in exts:
            src = base + '.' + ext
            if os.path.isfile(src):
                return src
    return None

def extarget(target, deps, commands):
    """Expect recipe to contain the compilation command followed by removal of the object file."""
    if len(commands) != 2:
        raise RuntimeError('Improper recipe for target: %s : %s' % (target.s, ' '.join(deps)))
    link = parselink(commands[0].exp)
    rm = parserm(commands[1].exp)
    # Resolve sources
    srcs = map(findsrc, link.objs)
    if not all(srcs):
        printverbose('Could not locate source files for: %r' % (link,))
        return None
    return Example(link.out, srcs, link.libs)
    #print 'extarget', target, deps, [x.exp for x in commands]

def runtarget(target, deps, commands):
    print('PARSE TARGET:', [c.exp for c in commands])
    runt = target.s[3:] # We know that target starts with 'run'
    rm_pre = parserm(commands[0].exp)
    if rm_pre:
        commands = commands[1:]
    cmd = iter(commands[0].exp)
    token, isfunc = next(cmd)
    if token == '-' or token == '-@':
        token, isfunc = next(cmd)
    assert isfunc and isinstance(token, pymake.functions.VariableRef) and token.vname.s == 'MPIEXEC'
    token, isfunc = next(cmd)
    assert not isfunc
    args = iter(token.split())
    assert next(args) in ['-n', '-np']
    np = int(next(args))
    rune = os.path.split(next(args))[-1]
    if os.path.basename(rune) != runt.split('_')[0]:
        printverbose('Non-conforming run target or executable name: target=%s exe=%s' % (target.s, rune))
    runargs = list(itertools.takewhile(lambda x: x != '>', args))
    runargs = [arg for arg in runargs if arg != '\\']
    cmpname = None
    try:
        token, isfunc = next(cmd)
        output = None
        if isfunc and isinstance(token, pymake.functions.VariableRef) and token.vname.s == 'DIFF':
            token, isfunc = next(cmd)
            outpath = token.split()[0]
            outfile = os.path.split(outpath)[1]
            cmpname = os.path.splitext(outfile)[0]
    except StopIteration:
        pass
    return Run(name=runt, exe=rune, np=np, args=runargs, cmpname=cmpname)

statements = pymake.parser.parsefile('makefile')

statementiter = iter(statements)
examples = collections.OrderedDict()
runs = collections.OrderedDict()
while True:
    try:
        s = next(statementiter)
    except StopIteration:
        break
    if isinstance(s, pymake.parserdata.Rule):
        commands = []
        while True:
            x = next(statementiter)
            if isinstance(x, pymake.parserdata.Command):
                commands.append(x)
            else:
                statementiter = itertools.chain([x], statementiter)
                break
        target = s.targetexp
        deps = [x for x in s.depexp.s.split() if x != 'chkopts']
        if s.targetexp.s.startswith('ex'):
            ex = extarget(target, deps, commands)
            if ex is not None:
                examples[ex.out] = ex
        elif s.targetexp.s.startswith('runex'):
            run = runtarget(target, deps, commands)
            if runs.has_key(run.exe):
                runs[run.exe].append(run)
            else:
                runs[run.exe] = [run]

# Find automatically run test cases
mkvars = distutils.sysconfig.parse_makefile('makefile')
testreq = dict()
for var in mkvars:
    if var.startswith('TESTEXAMPLES_'):
        requires = [req for req in var.split('_')[1:] if req not in 'C FORTRAN 13 17 18'.split()] # ignore some insignificant requirements
        for run in mkvars[var].split():
            if run.startswith('run'):
                testreq[run[3:]] = requires

def listfmt(strlist):
    string = ' '.join(strlist)
    if strlist == string.split():
        return string
    else:
        return strlist

def exfmt(ex):
    if os.path.splitext(ex.srcs[0])[0] == ex.out:
        return "executable(%r, libs=%r)" % (listfmt(ex.srcs), listfmt(ex.libs))
    else:
        return 'executable(name=%r, srcs=%r, libs=%r)' % (ex.out, listfmt(ex.srcs), listfmt(ex.libs))

def runfmt(ex, run, req):
    assert ex.out == run.exe
    if run.name.split('_')[0] == run.exe:
        id = ''.join(itertools.dropwhile(lambda x:x != '_', iter(run.name)))[1:]
        if len(id) == 0:
            id = '1'
    else:
        id = '1'
    if run.np == 1:
        npfmt = ''
    else:
        npfmt = ', np=%d' % run.np
    argsfmt = ", args=%r" % listfmt(run.args)
    if run.cmpname == '%s_%s' % (run.exe, id):
        cmpfmt = ''
    else:
        cmpfmt = ", compare=%r" % (run.cmpname,)
    if req is None:
        req = ', auto=False'
    elif not req:
        req = ''
    else:
        req = ", requires=%r" % listfmt(req)
    return 'test(id=%(id)r%(npfmt)s%(argsfmt)s%(cmpfmt)s%(req)s)' % locals()

def main():
    ptestfile = os.path.realpath('ptest.py')
    with open(ptestfile, 'w') as fd:
        fd.write('#!/usr/bin/env python\n\n')
        fd.write('import sys, os\n')
        fd.write("sys.path.append(os.path.join(os.environ['PETSC_DIR'],'conf'))\n")
        fd.write('import petsc.test\n\n')
        fd.write('db = petsc.getdb(__file__)\n\n')
        for ex in examples.values():
            fd.write('%s = db.%s\n' % (ex.out,exfmt(ex)))
            for run in runs.get(ex.out, []):
                fd.write('%s.%s\n' % (ex.out, runfmt(ex, run, testreq.get(run.name, None)),))
            fd.write('\n')
        fd.write('\n')
        fd.write("if __name__ == '__main__':\n")
        fd.write('    petsc.test.main()\n')
        print('Wrote %s' % ptestfile)

if __name__ == '__main__':
    main()
