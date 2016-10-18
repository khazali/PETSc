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

tap_out ./ex2 ex2.tmp ex2.err
tap_out 'diff ex2.tmp output/ex2.out' diff-ex2.tmp diff-ex2.err


