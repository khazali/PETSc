#!/usr/bin/env groovy

pipeline {
  
  agent none
  
  stages {
    stage('Configure') {
      steps {
        parallel (
          "c-exodus-dbg-builder" : {
            node('frog') {
              checkout scm
              sh './configure --download-suitesparse --download-mumps --download-scalapack --download-chaco --download-ctetgen --download-exodusii --download-cmake --download-pnetcdf --download-generator --download-hdf5 --download-zlib=1 --download-metis --download-ml --download-netcdf --download-parmetis --download-triangle --with-cuda --with-shared-libraries PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=${WORKSPACE}'
            }
          },
          "cuda-double" : {
            node('frog') {
              checkout scm
              sh './configure --with-cuda=1 --with-precision=double --with-clanguage=c PETSC_ARCH=arch-cuda-double PETSC_DIR=${WORKSPACE}'
            }
          },
          "viennacl" : {
            node('frog') {
              checkout scm
              sh './configure --download-viennacl --with-opencl-include=/usr/local/cuda/include --with-opencl-lib=\"-L/usr/local/cuda/lib64 -lOpenCL\" PETSC_ARCH=arch-viennacl PETSC_DIR=${WORKSPACE}'
            }
          },
          "linux-cxx-cmplx-pkgs-64idx" : {
            node('fedora') {
              checkout scm
              sh './configure --with-cc=clang --with-fc=gfortran --with-cxx=clang++ --with-clanguage=cxx CXXFLAGS=\"-Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -Wno-deprecated\" --with-scalar-type=complex --with-64-bit-indices=1 --download-mpich=1 --download-cmake=1 --download-make=1 --download-metis=1 --download-parmetis=1 --download-pastix=1 --download-ptscotch=1 --download-superlu_dist=1 --with-cxx-dialect=C++11 --download-elemental=1 PETSC_ARCH=arch-linux-cxx-cmplx-pkgs-64idx PETSC_DIR=${WORKSPACE}'
            }
          },
          "linux-dbg-quad" : {
            node('fedora') {
              checkout scm
              sh './configure --with-debugging=1 --download-f2cblaslapack=1 --with-precision=__float128 PETSC_ARCH=arch-linux-dbg-quad PETSC_DIR=${WORKSPACE}'
            }
          },
          "linux-pkgs-dbg-ftn-interfaces" : {
            node('fedora') {
              checkout scm
              sh './configure --with-cc=clang --with-cxx=clang++ --with-fc=gfortran --with-debugging=1 --download-mpich=1 --download-fblaslapack=1 --download-hypre=1 --download-cmake=1 --download-metis=1 --download-parmetis=1 --download-ptscotch=1 --download-suitesparse=1 --download-triangle=1 --download-superlu=1 --download-superlu_dist=1 --download-scalapack=1 --download-strumpack=1 --download-mumps=1 --download-elemental=1 --with-cxx-dialect=C++11 --download-spai=1 --download-parms=1 --download-chaco=1 PETSC_ARCH=arch-linux-pkgs-dbg-ftn-interfaces PETSC_DIR=${WORKSPACE}'
            }
          }
        )
      }
    }

    stage('Make') {
      steps {
        parallel (
          "c-exodus-dbg-builder" : {
            node('frog') {
              sh 'make PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=${WORKSPACE} all'
            }
          },
          "cuda-double" : {
            node('frog') {
              checkout scm
              sh 'make PETSC_ARCH=arch-cuda-double PETSC_DIR=${WORKSPACE} all'
            }
          },
          "viennacl" : {
            node('frog') {
              checkout scm
              sh 'make PETSC_ARCH=arch-viennacl PETSC_DIR=${WORKSPACE} all'
            }
          },
          "linux-cxx-cmplx-pkgs-64idx" : {
            node('fedora') {
              checkout scm
              sh 'make PETSC_ARCH=arch-linux-cxx-cmplx-pkgs-64idx PETSC_DIR=${WORKSPACE} all'
            }
          },
          "linux-dbg-quad" : {
            node('fedora') {
              checkout scm
              sh 'make PETSC_ARCH=arch-linux-dbg-quad PETSC_DIR=${WORKSPACE} all'
            }
          },
          "linux-pkgs-dbg-ftn-interfaces" : {
            node('fedora') {
              sh 'make PETSC_ARCH=arch-linux-pkgs-dbg-ftn-interfaces PETSC_DIR=${WORKSPACE} all'
            }
          }
        )
      }
    }

    stage('Examples') {
      steps {
        parallel (
          "c-exodus-dbg-builder" : {
            node('frog') {
              sh 'make PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=${WORKSPACE} -f gmakefile test'
            }
          },
          "cuda-double" : {
            node('frog') {
              checkout scm
              sh 'make PETSC_ARCH=arch-cuda-double PETSC_DIR=${WORKSPACE} -f gmakefile test'
            }
          },
          "viennacl" : {
            node('frog') {
              checkout scm
              sh 'make PETSC_ARCH=arch-viennacl PETSC_DIR=${WORKSPACE} -f gmakefile test'
            }
          },
          "linux-cxx-cmplx-pkgs-64idx" : {
            node('fedora') {
              checkout scm
              sh 'make PETSC_ARCH=arch-linux-cxx-cmplx-pkgs-64idx PETSC_DIR=${WORKSPACE} -f gmakefile test'
            }
          },
          "linux-dbg-quad" : {
            node('fedora') {
              checkout scm
              sh 'make PETSC_ARCH=arch-linux-dbg-quad PETSC_DIR=${WORKSPACE} -f gmakefile test'
            }
          },
          "linux-pkgs-dbg-ftn-interfaces" : {
            node('fedora') {
              sh 'make PETSC_ARCH=arch-linux-pkgs-dbg-ftn-interfaces PETSC_DIR=${WORKSPACE} -f gmakefile test'
            }
          }
        )
      }
    }
  }
}
