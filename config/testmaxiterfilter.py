#!/usr/bin/env python
''' Filter a stream by taking tuples of regular expressions and lists maximum iteration counts and replacing all
integers less than the maximum values with the maximum values

e.g.

${PETSC_DIR}/config/testmaxiter.py "SNES iterations = " 7,7,5

Will convert

    SNES iterations = 1
    SNES iterations = 7
    SNES iterations = 6

to

    SNES iterations = [<= 7]
    SNES iterations = [<= 7]
    SNES iterations = 6
'''

if __name__ == '__main__':
    import sys,re,fileinput

    regexes = []
    maxiters = []
    # first arguments are prefix regexes
    for arg in sys.argv[1::2]:
        regexes.append(re.compile('(' + arg+r')([0-9]+)'))
    # second arguments are lists of maxima
    for arg in sys.argv[2::2]:
        maxiters.append([int(s) for s in arg.split(',')])
    rules = zip(regexes,maxiters)
    # test each rule on each line
    for line in sys.stdin:
        for rule in rules:
            match = rule[0].search(line)
            if match:
                val = int(match.group(2))
                maxiter = rule[1].pop(0)
                if val <= maxiter:
                    line = rule[0].sub(r'\1[<= ' + str(maxiter) + ']',line)
        sys.stdout.write(line)




