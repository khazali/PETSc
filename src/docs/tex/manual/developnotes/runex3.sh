#!/bin/sh
# This will be substituted in by gmakegentest.py which can do a global
# count info
global_count=6

. petsc_harness.sh

for i in aij baij sbaij; do
  for j in 2 3; do
    petsc_testrun "./ex3 -f ${DATAFILESPATH}/matrices/small -mat_type $i -matload_block_size $j" ex3.tmp ex3.err
    petsc_testrun 'diff ex3.tmp output/ex3.out' diff-ex1.tmp diff-ex1.err
  done
done

petsc_testend


