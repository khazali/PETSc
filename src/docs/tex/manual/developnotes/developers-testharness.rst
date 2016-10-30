
PETSc Testing System
===============================

The PETSc test system consists of:

  1. A language contained within the source files that describes the
     tests to be run
  2. The *test generator* (`config/gmakegentest.py`) that at the 
     `make` step parses the source files and generates the makefiles 
     and shell scripts that compose:
  3. The *petsc test harness*: a harness consisting of makefile and
     shell scripts that runs the executables with several
     logging and reporting features. 


PETSc Test description language
-------------------------------

PETSc tests and tutorials contain within their file a simple language to 
describe tests and subtests required to run executables associated with
compilation of that file.  The general skeleton of the file is::

      static char help[] = "A simple MOAB example\n\

      /*T
         Concepts: Demonstrate MOAB bindings for DM
         requires: moab
      T*/
      
      ...
      <source code>
      ...

      /*TESTS
         test:
           suffix: 1
           requires: !complex
         test:
           suffix: 2
           args: -debug -fields v1,v2,v3 
      TESTS*/

For our language, a *test* is associated with a shell script and
makefile target.  A *subtest* is a command invoked within the test
script.  A typical test will have two subtests: one to execute the 
executable and one to compare the output with the *expected results*.
For example::

      mpiexec -n 1 ../ex1 1> ex1.tmp 2> ex1.err
      diff ex1.tmp output/ex1.out 1> diff-ex1.tmp 2> diff-ex1.err

In practice, we want to do various logging and counting by the test
harness, but this is explained further below.  The input language
supports a simple, yet flexible, tests control and we begin by
describing this language.


Runtime language options
~~~~~~~~~~~~~~~~~~~~~~~~~~

At the end of each test, a marked comment block that uses YAML is
inserted that describes the test to be run.  The elements of the
test are done with a set of supported key words that sets up the test.

The goals of the language are to:
 1. Be as minimal as possible with the simplest test requiring only one
    keyword
 2. Be independent of the filename such that a file can be renamed
    without rewriting the tests
 3. Be intuitive

To enable the second bullet, the *basestring* of the filename is defined
as the filename without the extention; i.e., if the filename is `ex1.c`
then `basestring=ex1`.f 

With this background, these keywords are are:

 + test: (**Required**)
     - This is the top level keyword for the tests.  All other are
       subsets of this keyword

 +  suffix: (**Optional**; *Default:* `suffix=''`)
     - The testname is given by: `testname='run'+basestring`
        if suffix is set to an empty string, and by 
        `testname='run'+basestring+'_'+suffix`
     - This can only be specified for top level test nodes

 + output_file: (**Optional**; *Default:* `output_file=testname+'.out`)
     - The output of the test is to be compared to an *expected result*
       whose name is given by output_file.  
     - This file is described relative to the source directory of the 
       source file and should be in the output subdirectory (e.g.,
       `output/ex1.out`)

 + nsize: (**Optional**; *Default:* `nsize=1`)
     - The integer that is passed to mpiexec; i.e., `mpiexec -n nsize`

 + args: (**Optional**; *Default:* `""`)
     - The arguments to pass to the executable

 + filter: (**Optional**; *Default:* `""`)
     - Sometimes only a subset of the output is meant to be tested
       against the expected result.  If this keyword is used, it 
       processes the executable output and puts it into the file
       to be actually compared with output_file.
     - The value of this is the command to be run; e.g., `grep foo` or
       `sort -nr`
     - A skeleton example of the resultant commands to be run is::

           mpiexec -n 1 ../ex1 1> ex1.tmp 2> ex1.err
           grep residual ex1.tmp > grep-ex1.tmp
           diff grep-ex1.tmp output/ex1.out 1> diff-ex1.tmp 2> diff-ex1.err

       In practice, each of these steps become a subtest for reporting.

 + requires: (**Optional**; *Default:* `""`)
     -  A space-delimited list of requirements of run requirements (not
        build requirements. See Build requirements below)
     - In general, the language supports `and` and `not` constructs
       using `! => not` and `, => and`
     - MPIUNI should work for all -n 1 examples so this need not be in the requirements list
     - Inputs sometimes include external matrices that are found in the
       DATAFILES path.  `requires: DATAFILES` can be specifed for these
       tests.
     - Packages are specified with lower case specification; e.g.,
       `requires: superlu_dist`
     - Any defined variable in petscconf.h can be specified with the
       `defined(...)` syntax; e.g., `defined(PETSC_USE_INFO)`

Additional specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the above keywords, other language features are
supported:

 + for loops:  Specifying `{{ ... }}` will create for loops over
   enclosed space-delmited list.  For loops are supported within nsize
   and args.  An example would be::

             args: -matload_block_size {{2,3}}

   In this case, two execution lines would be addded with two different
   arguments.  Associated `diff` lines would be added as well
   automatically.  See examples below for how it works in practice.


Test block examples
~~~~~~~~~~~~~~~~~~~~

This is the simplest test block::

      /*TESTS
        test: 
      TESTS*/

If this block is in ex1.c, then it will create a `runex1` test that
requires only one processor/thread, with no arguments, and diff the
resultant output with `output/ex1.out`.

For fortran, the equivalent is::

      !/*TESTS
      !  test: 
      !TESTS*/

A fuller example would be::
  
      /*TESTS
        test: 
        test:
          suffix: 1
          nsize: 2
          args:  -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type gmres -ksp_gmres_cgs_refinement_type refine_always -s2_ksp_type bcgs -s2_pc_type jacobi -s2_ksp_monitor_short
          requires: x
      */TESTS

This creates two tests.  Assuming that this is `ex1.c`, the tests would
be `runex1` and `runex1_1`.  

An example using a for loop would be::

      /*TESTS
        test:
             suffix: 1
             args:   -f ${DATAFILESPATH}/matrices/small -mat_type aij
             requires: datafilespath
         test:
             suffix: 2
             output_file: output/ex138.out
             args: -f ${DATAFILESPATH}/matrices/small -mat_type baij -matload_block_size {{2,3}}
             requires: datafilespath
      */TESTS

In this example, runex138_2 will invoke ex138 twice with two different
arguments, but both are diffed with the same file.  

An example for showing the hieararchial nature of the test specification is::

      test: 
        suffix:2
        output_file: output/ex1.out
        args: -f ${DATAFILESPATH}/matrices/small -mat_type baij
        test:
             args: -matload_block_size 2
        test:
             args: -matload_block_size 3


This is functionally equivalent to the for loop shown above.
If you have different output files, this example is more extensible
however as the different output_files can be placed under tests.


Build language options
~~~~~~~~~~~~~~~~~~~~~~~~


It is possible to specify issues related to the compilation of the
source file.  The language is:

 + requires: (**Optional**; *Default:* `""`)
    1. Same as the runtime requirements (e.g., can include requires: fftw)
       but also requirements related to types:
       A. Precision types: single, double, quad, int32
       B. Scalar types: complex  (and !complex)
 + depends: (**Optional**; *Default:* `""`)
    1. List any dependencies required to compile the file


A typical example for compiling for real/double only is::

      /*T
        requires: !complex
      T*/



PETSC Test Harness
--------------------------

The goals of the PETSc Test Harness are to:

  1. Provide standard output used by other testing tools
  2. Lightweight as possible and easily fit within the PETSc build chain
  3. Provide information on all tests, even those that are not built or
     run because they do not meet the configuration requirements

Before understanding the test harness, it is first important to
understand the desired requirements for reporting and logging.

Test output standards: TAP
==========================

The PETSc test system is designed to be compliant with the Test Anything
Protocal (TAP): See https://testanything.org/tap-specification.html

This is a very simple standard designed to allow testing tools to work
together easily.  There are libraries to enable the output to be used
easily including sharness, which is used by the git team.  However, the
simplicity of the petsc tests and TAP specification means that we use
our own simple harness given by a single shell script that each file
sources: `petsc_harness.sh`.

As an example, consider this test input::

     test:
         suffix: 2
         output_file: output/ex138.out
         args: -f ${DATAFILESPATH}/matrices/small -mat_type {{aij,baij,sbaij}} -matload_block_size {{2,3}}
         requires: datafilespath

A sample output would be::

      ok 1 In mat...tests: "./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 2"
      ok 2 In mat...tests: "Diff of ./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 2"
      ok 3 In mat...tests: "./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 3"
      ok 4 In mat...tests: "Diff of ./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 3"
      ok 5 In mat...tests: "./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type baij -matload_block_size 2"
      ok 6 In mat...tests: "Diff of ./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type baij -matload_block_size 2"
      ...

      ok 11 In mat...tests: "./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type saij -matload_block_size 2"
      ok 12 In mat...tests: "Diff of ./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 2"


Test harness implementation
============================

Most of the requirements for being TAP-compliant lie in the shell
scripts so we focus on that description.  

A sample shell script is given by::

      #!/bin/sh
      . petsc_harness.sh

      petsc_testrun ./ex1 ex1.tmp ex1.err
      petsc_testrun 'diff ex1.tmp output/ex1.out' diff-ex1.tmp diff-ex1.err

      petsc_testend

`petsc_harness.sh` is a small shell script that provides the logging and
reporting functions `petsc_testrun` and `petsc_testend`.

A small sample of the output from the test harness would be::

      ok 1 ./ex1
      ok 2 diff ex1.tmp output/ex1.out
      not ok 4 ./ex2
      #	ex2: Error: cannot read file
      not ok 5 diff ex2.tmp output/ex2.out
      ok 7 ./ex3 -f /matrices/small -mat_type aij -matload_block_size 2
      ok 8 diff ex3.tmp output/ex3.out
      ok 9 ./ex3 -f /matrices/small -mat_type aij -matload_block_size 3
      ok 10 diff ex3.tmp output/ex3.out
      ok 11 ./ex3 -f /matrices/small -mat_type baij -matload_block_size 2
      ok 12 diff ex3.tmp output/ex3.out
      ok 13 ./ex3 -f /matrices/small -mat_type baij -matload_block_size 3
      ok 14 diff ex3.tmp output/ex3.out
      ok 15 ./ex3 -f /matrices/small -mat_type sbaij -matload_block_size 2
      ok 16 diff ex3.tmp output/ex3.out
      ok 17 ./ex3 -f /matrices/small -mat_type sbaij -matload_block_size 3
      ok 18 diff ex3.tmp output/ex3.out
      # FAILED   4 5
      # failed 2/16 tests; 87.500% ok
