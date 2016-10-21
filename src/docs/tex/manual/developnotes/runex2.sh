#!/bin/sh
# This will be substituted in by gmakegentest.py which can do a global
# count info
global_count=4

. petsc_harness.sh

petsc_testrun ./ex2 ex2.tmp ex2.err
petsc_testrun 'diff ex2.tmp output/ex2.out' diff-ex2.tmp diff-ex2.err

petsc_testend

