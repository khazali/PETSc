#!/bin/sh
if test ${#@}>0; then
  count=$1
else
  count=0
fi
function tap_out() {
  # First arg = Basic command
  # Second arg = stdout file
  # Third arg = stderr file
  let count=$count+1
  cmd="$1 > $2 2> $3"
  eval $cmd
  if test $? == 0; then
      printf "ok $count $1\n"
  else
      printf "not ok $count $1\n"
      awk '{print "#\t" $0}' < $3
  fi
}

for i in aij baij sbaij; do
  for j in 2 3; do
    tap_out "./ex3 -f ${DATAFILESPATH}/matrices/small -mat_type $i -matload_block_size $j" ex3.tmp ex3.err
    tap_out 'diff ex3.tmp output/ex3.out' diff-ex1.tmp diff-ex1.err
  done
done



