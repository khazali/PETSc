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

tap_out ./ex1 ex1.tmp ex1.err
tap_out 'diff ex1.tmp output/ex1.out' diff-ex1.tmp diff-ex1.err


