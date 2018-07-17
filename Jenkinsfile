#!/usr/bin/env groovy

pipeline {
  agent none
  stages {
    stage('Configure') {
        steps {
            parallel (
                "arch-c-exodus-dbg-builder" : {
                  node('frog') {
                    checkout scm
                    sh './configure --download-suitesparse --download-mumps --download-scalapack --download-chaco --download-ctetgen --download-exodusii --download-cmake --download-pnetcdf --download-generator --download-hdf5 --download-zlib=1 --download-metis --download-ml --download-netcdf --download-parmetis --download-triangle --with-cuda --with-shared-libraries PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=/sandbox/petsc/workspace/PETSc'
                  }
                },
                "arch-linux-pkgs-dbg-ftn-interfaces" : {
                  node('n-gage') {
                    checkout scm
                    sh './configure --with-cc=clang --with-cxx=clang++ --with-fc=gfortran --with-debugging=1 --download-mpich=1 --download-fblaslapack=1 --download-hypre=1 --download-cmake=1 --download-metis=1 --download-parmetis=1 --download-ptscotch=1 --download-suitesparse=1 --download-triangle=1 --download-superlu=1 --download-superlu_dist=1 --download-scalapack=1 --download-strumpack=1 --download-mumps=1 --download-elemental=1 --with-cxx-dialect=C++11 --download-spai=1 --download-parms=1 --download-chaco=1 PETSC_ARCH=arch-linux-pkgs-dbg-ftn-interfaces PETSC_DIR=/sandbox/petsc/workspace/PETSc'
                  }
                }
            )
        }
    }
    
    stage('Make') {
        steps {
            parallel (
                "arch-c-exodus-dbg-builder" : {
                  node('frog') {
                    sh 'make PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=/sandbox/petsc/workspace/PETSc all'
                  }
                },
                "arch-linux-pkgs-dbg-ftn-interfaces" : {
                  node('n-gage') {
                    sh 'make PETSC_ARCH=arch-linux-pkgs-dbg-ftn-interfaces PETSC_DIR=/sandbox/petsc/workspace/PETSc all'
                  }
                }
            )
        }
    }
    
    stage('Examples') {
      steps {
          parallel (
              "arch-c-exodus-dbg-builder" : {
                node('frog') {
                  sh "make PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=/sandbox/petsc/workspace/PETSc -f gmakefile test search='tao_unconstrained%'"
                }
              },
              "arch-linux-pkgs-dbg-ftn-interfaces" : {
                node('n-gage') {
                  sh "make PETSC_ARCH=arch-linux-pkgs-dbg-ftn-interfaces PETSC_DIR=/sandbox/petsc/workspace/PETSc -f gmakefile test search='tao_unconstrained%'"
                }
              }
          )
      }
    }
  }
}
