



PETSc Test description language
===============================

Introduction
-------------

PETSc tests and tutorials contain within their file a simple to 
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

      /*TEST
         test_suffix: 1
           requires: !complex
         test_suffix: 2
           args: -debug -fields v1,v2,v3 
      TEST*/

For our language, a *test* is associated with a shell script and
makefile target.  A *subtest* is a command invoked within the test
script.  A typical test will have two subtests: one to execute the 
executable and one to compare the output with the *expected results*.
For example::

      petsc_test "mpiexec -n 1 ../ex1" ex1.tmp ex1.err
      petsc_test "diff ex1.tmp output/ex1.out " diff-ex1.tmp diff-ex1.err

Our input language supports a simple, yet flexible, tests/subtests control as
described below.


Runtime language options
--------------------------

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

 + test_suffix: (**Required**)
     - The testname is given by: `testname='run'+basestring`
        if output_suffix is set to an empty string, and by 
        `testname='run'+basestring+'_'+output_suffix`
     - This is the top level keyword for the tests.  All other are
       subsets of this keyword
     - A special keyword value can be specified: `subtest` for
       controlling subtests of the main test.
       This is explained in more detail below

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

 + grep: (**Optional**; *Default:* `""`)
     - Sometimes only a subset of the output is meant to be tested
       against the expected result.  If this keyword is used, it 
       processes the output and puts it into output file

 + sort: (**Optional**; *Default:* `""`)
     -  Similar to grep, this allows sorting of the output to compare to
        the expected results

 + requires: (**Optional**; *Default:* `""`)
     -  A comma-delimited list of requirements of run requirements (not
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
   enclosed comma-delmited list.  For loops are supported within nsize
   and args.  An example would be::

             args: -matload_block_size {{2,3}}

   In this case, two execution lines would be addded with two different
   arguments.  Associated `diff` lines would be added as well
   automatically.  See examples below for how it works in practice.

test_suffix: subtest option
~~~~~~~~~~~~~~~~~~~~~~~~~~~

As defined in the introduction to the test syystemp, a "test" is defined
as that given by 1 shell script/makefile target, and a subtest is
defined as a command executed within that shell script, with a typical
example containing two examples.  A for loop is an example of generating
more subtests.  For example, a for loop over two variables will generate
four subtests in a typical example.

The test suffix variable can specify how the subtests are grouped.
We consider these examples to illustrate::

      test_suffix: 1
        output_file: output/ex1.out
        args=-f ${DATAFILESPATH}/matrices/small
        test_suffix: subtest
             args: -matload_block_size 2
        test_suffix: subtest
             args: -matload_block_size 3

This example is equivalent to the for loop above.  A
single test is created (e.g., `runex1_1`) but additional
subtests are specified manually.  Both tests use the same
input, and compare against the same output file.

Here is an example of the inverse of the this example::

      test_suffix: subtest
        output_file: output/ex1.out
        args=-f ${DATAFILESPATH}/matrices/small
        test_suffix: 1
             args: -matload_block_size 2
        test_suffix: 2
             args: -matload_block_size 3

Here instead of the different `matload_block_size` changes
being subtest variations, these subtests will be placed into
separate scripts: (e.g., `runex1_1` and `runex1_2`).

Finally, as comparison, consider this block::

      test_suffix: 1
        output_file: output/ex1.out
        args=-f ${DATAFILESPATH}/matrices/small
        test_suffix: 2
             args: -matload_block_size 2
        test_suffix: 2
             args: -matload_block_size 3

This block will generate 3 tests (e.g., `runex1_1`, `runex1_2`, and
`runex1_3`) with each test have 2 subtests.  All will use the same
output_file for the tested results.


Test block examples
--------------------

This is the simplest test block::

      /*TEST
        test_suffix: 
      TEST*/

If this block is in ex1.c, then it will create a `runex1` test that
requires only one processor/thread, with no arguments, and diff the
resultant output with `output/ex1.out`.

For fortran, the equivalent is::

      !/*TEST
      !  test_suffix: 
      !TEST*/

A fuller example would be::
  
      /*TEST
        test_suffix: 
        test_suffix: 1
          nsize: 2
          args:  -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type gmres -ksp_gmres_cgs_refinement_type refine_always -s2_ksp_type bcgs -s2_pc_type jacobi -s2_ksp_monitor_short
          requires: x
      */TEST

This creates two tests.  Assuming that this is `ex1.c`, the tests would
be `runex1` and `runex1_1`.  

An example using a for loop would be::

      /*TEST
        test_suffix: 1
             args:   -f ${DATAFILESPATH}/matrices/small -mat_type aij
             requires: datafilespath
         test_suffix: 2
             output_file: output/ex138.out
             args: -f ${DATAFILESPATH}/matrices/small -mat_type baij -matload_block_size {{2,3}}
             requires: datafilespath
      */TEST


In this example, runex138_2 will invoke ex138 twice with two different
arguments, but both are diffed with the same file.  


Build language options
------------------------


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

     test_suffix: 2
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

