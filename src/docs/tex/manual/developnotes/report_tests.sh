#!/bin/sh
total=0
success=0
failed=0
failures=""
for file in counts/*.counts; do
  total_str=`grep total $file | cut -f2 -d" "`
  let total=$total+$total_str
  failed_str=`grep failed $file | cut -f2 -d" "`
  let failed=$failed+$failed_str
  success_str=`grep success $file | cut -f2 -d" "`
  let success=$success+$success_str
  failures_fromc=`grep failures $file | cut -f2- -d" "`
  if test -n "${failures_fromc//[[:space:]]}"; then
    failures="$failures $failures_fromc"
  fi
done
percent=`echo "scale=3; ${success}/${total} * 100" | bc -l`
echo "# FAILED $failures"
echo "# failed ${failed}/${total} tests; ${percent}% ok"
