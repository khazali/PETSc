#!/bin/sh

# These are created by gmakegentest.py
mpiexec='./mpiexec -n '
nsize=1
args=""
output_file=ex1.out
testname=ex1

# count info
global_count=0

. petsc_harness.sh

petsc_testrun "${mpiexec} ${nsize} ./ex1 ${args}" ${testname}.tmp ${testname}.err
petsc_testrun "diff ${testname}.tmp output/${output_file}" diff-${testname}.out diff-${testname}.out

petsc_testend


