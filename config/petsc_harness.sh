

scriptname=`basename $0`
rundir=${scriptname%.sh}
TIMEOUT=60

if test "$PWD"!=`dirname $0`; then
  cd `dirname $0`
fi
if test -d "${rundir}" && test -n "${rundir}"; then
  rm -f ${rundir}/*.tmp ${rundir}/*.err ${rundir}/*.out
fi
mkdir -p ${rundir}
if test -n "${runfiles}"; then
  for runfile in ${runfiles}; do
      subdir=`dirname ${runfile}`
      mkdir -p ${rundir}/${subdir}
      cp -r ${runfile} ${rundir}/${subdir}
  done
fi
cd ${rundir}

#
# Method to print out general and script specific options
#
print_usage() {

cat >&2 <<EOF
Usage: $0 [options]

OPTIONS
  -a <args> ......... Override default arguments
  -c <cleanup> ...... Cleanup (remove generated files)
  -d ................ Launch in debugger
  -e <args> ......... Add extra arguments to default
  -f ................ force attempt to run test that would otherwise be skipped
  -h ................ help: print this message
  -n <integer> ...... Override the number of processors to use
  -j ................ Pass -j to petscdiff (just use diff)
  -J <arg> .......... Pass -J to petscdiff (just use diff with arg)
  -m ................ Update results using petscdiff
  -M ................ Update alt files using petscdiff
  -t ................ Override the default timeout (default=$TIMEOUT sec)
  -V ................ run Valgrind
  -v ................ Verbose: Print commands
EOF

  if declare -f extrausage > /dev/null; then extrausage; fi
  exit $1
}
###
##  Arguments for overriding things
#
verbose=false
cleanup=false
debugger=false
force=false
diff_flags=""
while getopts "a:cde:fhjJ:mMn:t:vV" arg
do
  case $arg in
    a ) args="$OPTARG"       ;;  
    c ) cleanup=true         ;;  
    d ) debugger=true        ;;  
    e ) extra_args="$OPTARG" ;;  
    f ) force=true           ;;
    h ) print_usage; exit    ;;  
    n ) nsize="$OPTARG"      ;;  
    j ) diff_flags="-j"      ;;  
    J ) diff_flags="-J $OPTARG" ;;  
    m ) diff_flags="-m"      ;;  
    M ) diff_flags="-M"      ;;  
    t ) TIMEOUT=$OPTARG      ;;  
    V ) mpiexec="petsc_mpiexec_valgrind $mpiexec" ;;  
    v ) verbose=true         ;;  
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

# Individual tests can extend the default
export MPIEXEC_TIMEOUT=$((TIMEOUT*timeoutfactor))
STARTTIME=`date +%s`

if test -n "$extra_args"; then
  args="$args $extra_args"
fi
if $debugger; then
  args="-start_in_debugger $args"
fi


# Init
success=0; failed=0; failures=""; rmfiles=""
total=0
todo=-1; skip=-1
job_level=0

function petsc_testrun() {
  # First arg = Basic command
  # Second arg = stdout file
  # Third arg = stderr file
  # Fourth arg = label for reporting
  # Fifth arg = Filter
  rmfiles="${rmfiles} $2 $3"
  tlabel=$4
  filter=$5
  cmd="$1 > $2 2> $3"
  if test -n "$filter"; then
    if test "${filter:0:6}"=="Error:"; then
      filter=${filter##Error:}
      cmd="$1 2>&1 | cat > $2"
    fi
  fi
  echo "$cmd" > ${tlabel}.sh; chmod 755 ${tlabel}.sh

  eval "{ time -p $cmd ; } 2>> timing.out"
  cmd_res=$?
  touch "$2" "$3"
  # ETIMEDOUT=110 on most systems (used by Open MPI 3.0).  MPICH uses
  # 255.  Earlier Open MPI returns 1 but outputs about MPIEXEC_TIMEOUT.
  if [ $cmd_res -eq 110 -o $cmd_res -eq 255 ] || \
        fgrep -q -s 'APPLICATION TIMED OUT' "$2" "$3" || \
        fgrep -q -s MPIEXEC_TIMEOUT "$2" "$3" || \
        fgrep -q -s 'APPLICATION TERMINATED WITH THE EXIT STRING: job ending due to timeout' "$2" "$3" || \
        grep -q -s "Timeout after [0-9]* seconds. Terminating job" "$2" "$3"; then
    timed_out=1
    # If timed out, then ensure non-zero error code
    if [ $cmd_res -eq 0 ]; then
      cmd_res=1
    fi
  fi

  # Handle filters separately and assume no timeout check needed
  if test -n "$filter"; then
    cmd="cat $2 | $filter > $2.tmp 2>> $3 && mv $2.tmp $2"
    echo "$cmd" >> ${tlabel}.sh
    eval "$cmd"
  fi

  # Report errors
  if test $cmd_res == 0; then
    if "${verbose}"; then
     printf "ok $tlabel $cmd\n" | tee -a ${testlogfile}
    else
     printf "ok $tlabel\n" | tee -a ${testlogfile}
    fi
    let success=$success+1
  else
    if "${verbose}"; then 
      printf "not ok $tlabel $cmd\n" | tee -a ${testlogfile}
    else
      printf "not ok $tlabel\n" | tee -a ${testlogfile}
    fi
    if [ -n "$timed_out" ]; then
      printf "#\tExceeded timeout limit of $MPIEXEC_TIMEOUT s\n" | tee -a ${testlogfile}
    else
      # We've had tests fail but stderr->stdout. Fix with this test.
      if test -s $3; then
        awk '{print "#\t" $0}' < $3 | tee -a ${testlogfile}
      else
        awk '{print "#\t" $0}' < $2 | tee -a ${testlogfile}
      fi
    fi
    let failed=$failed+1
    failures="$failures $tlabel"
  fi
  let total=$success+$failed
  return $cmd_res
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
  ENDTIME=`date +%s`
  timing=`touch timing.out && egrep '(user|sys)' timing.out | awk '{sum += sprintf("%.2f",$2)} END {printf "%.2f\n",sum}'`
  printf "time $timing\n" >> $logfile
  if $cleanup; then
    echo "Cleaning up"
    /bin/rm -f $rmfiles
  fi
}

function petsc_mpiexec_valgrind() {
  mpiexec=$1;shift
  npopt=$1;shift
  np=$1;shift

  valgrind="valgrind -q --tool=memcheck --leak-check=yes --num-callers=20 --track-origins=yes --suppressions=$petsc_bindir/maint/petsc-val.supp"

  $mpiexec $npopt $np $valgrind $*
}
export LC_ALL=C
