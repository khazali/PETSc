
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
total=0
todo=-1; skip=-1

function petsc_testrun() {
  # First arg = Basic command
  # Second arg = stdout file
  # Third arg = stderr file
  # Fourth arg = label for reporting
  # Fifth arg = Filter
  rmfiles="${rmfiles} $2 $3"
  let count=$count+1
  tlabel=$4
  filter=$5

  if test -z "$filter"; then
    cmd="$1 > $2 2> $3"
  else
    cmd="$1 | $filter > $2 2> $3"
  fi
  eval $cmd
  if test $? == 0; then
      printf "ok $count $tlabel\n"
      let success=$success+1
  else
      printf "not ok $count $tlabel\n"
      awk '{print "#\t" $0}' < $3
      let failed=$failed+1
      failures="$failures $count"
  fi
  let total=$success+$failed
}

function petsc_testend() {
  logfile=$1/counts/${label}.counts
  logdir=`dirname $logfile`
  if ! test -d "$logdir"; then
    mkdir -p $logdir
  fi
  if ! test -e "$logfile"; then
    touch $logfile
  fi
  printf "total $total\n" > $logfile
  printf "success $success\n" >> $logfile
  printf "failed $failed\n" >> $logfile
  printf "failures $failures\n" >> $logfile
  if test ${todo} -gt 0; then
    printf "todo $todo\n" >> $logfile
  fi
  if test ${skip} -gt 0; then
    printf "skip $skip\n" >> $logfile
  fi
  if $cleanup; then
     /bin/rm -f $rmfiles
  fi
}
