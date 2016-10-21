
while getopts "a:e:m:n:o:t:" arg
do
  case $arg in
    a ) args=$OPTARG     ;;  
    e ) extra_args=$OPTARG     ;;  
    n ) nsize=$OPTARG     ;;  
    o ) output_file=$OPTARG     ;;  
    t ) testname=$OPTARG     ;;  
    *)  # To take care of any extra args
      if test -n "$OPTARG"; then
        eval $arg=\"$OPTARG\"
      else
        eval $arg=found
      fi
      ;;
  esac
done
shift $(( $OPTIND - 1 ))

if test -n "$extra_args"; then
  args="$args $extra_args"
fi

# Init
cleanup=False
success=0; failed=0; failures=""; rmfiles=""

function petsc_testrun() {
  # First arg = Basic command
  # Second arg = stdout file
  # Third arg = stderr file
  rmfiles="${rmfiles} $2 $3"
  let count=$count+1

  cmd="$1 > $2 2> $3"
  eval $cmd
  if test $? == 0; then
      printf "ok $count $1\n"
      let success=$success+1
  else
      printf "not ok $count $1\n"
      awk '{print "#\t" $0}' < $3
      let failed=$failed+1
      failures="$failures $count"
  fi
}

function petsc_testend() {
  logfile="counts/"${0}".counts"
  let total=$success+$failed
  echo "total $total" > $logfile
  echo "success $success" >> $logfile
  echo "failed $failed" >> $logfile
  echo "failures $failures" >> $logfile
  if $cleanup; then
     /bin/rm -f $rmfiles
  fi
}
