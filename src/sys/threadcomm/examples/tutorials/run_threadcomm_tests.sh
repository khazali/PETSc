#!/bin/sh

nthreads=4

echo ""
echo "Run ex6 test"
make ex6
./ex6 -n 1000 -threadcomm_type openmp -threadcomm_model user -threadcomm_nthreads $nthreads

echo ""
echo "Run ex7 test"
make ex7
./ex7 -n 1000 -threadcomm_type pthread -threadcomm_model auto -threadcomm_nthreads $nthreads

echo ""
echo "Run ex8 test pthread"
make ex8
./ex8 -n 1000 -threadcomm_type pthread -threadcomm_model loop -threadcomm_nthreads $nthreads

echo ""
echo "Run ex8 test openmp"
make ex8
./ex8 -n 1000 -threadcomm_type openmp -threadcomm_model loop -threadcomm_nthreads $nthreads

echo ""
echo "Run ex8 test nothread"
make ex8
./ex8 -n 1000

echo ""
echo "Run ex10 test"
make ex10
./ex10 -n 10 -threadcomm_type openmp -threadcomm_model user -threadcomm_nthreads $nthreads

echo ""
echo "Run ex11 test"
make ex11
./ex11 -n 1000 -threadcomm_type pthread -threadcomm_model user -threadcomm_nthreads $nthreads
