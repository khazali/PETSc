
header="""#!/bin/sh

# These are created by gmakegentest.py
mpiexec='@MPIEXEC@'
exec=@EXEC@
testname='@TESTNAME@'

# count info
global_count=0

. @TESTROOT@/petsc_harness.sh
"""

datfilepath="@DATAFILEPATH@"
footer="petsc_testend"

todoline="echo ok # TODO @TODOCOMMENT@"
skipline="echo not ok # SKIP @SKIPCOMMENT@"
mpitest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} @ARGS@" @REDIRECT_FILE@ ${testname}.err @FILTER@'
difftest='petsc_testrun "diff @REDIRECT_FILE@ @OUTPUT_FILE@" diff-${testname}.out diff-${testname}.out ""'
commandtest='petsc_testrun "@COMMAND@" @REDIRECT_FILE@ ${testname}.err @FILTER@'
