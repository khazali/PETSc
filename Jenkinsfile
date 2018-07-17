pipeline {
  agent none
  
  stages {
    
    stage('Configure') {
      parallel(
        'frog' : {
          node('frog') {
            sh './configure --download-suitesparse --download-mumps --download-scalapack --download-chaco --download-ctetgen --download-exodusii --download-cmake --download-pnetcdf --download-generator --download-hdf5 --download-zlib=1 --download-metis --download-ml --download-netcdf --download-parmetis --download-triangle --with-cuda --with-shared-libraries PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=/sandbox/petsc/workspace/PETSc'
          }
        }
        'n-gage' : {
          node('n-gage') {
            sh './configure --with-cc=clang --with-cxx=clang++ --with-fc=gfortran --with-debugging=1 --download-mpich=1 --download-fblaslapack=1 --download-hypre=1 --download-cmake=1 --download-metis=1 --download-parmetis=1 --download-ptscotch=1 --download-suitesparse=1 --download-triangle=1 --download-superlu=1 --download-superlu_dist=1 --download-scalapack=1 --download-strumpack=1 --download-mumps=1 --download-elemental=1 --with-cxx-dialect=C++11 --download-spai=1 --download-parms=1 --download-chaco=1 PETSC_ARCH=arch-linux-pkgs-dbg-ftn-interfaces PETSC_DIR=/sandbox/petsc/workspace/PETSc'
          }
        }
      )
    }
    
    stage('Make') {
      parallel(
        'frog' : {
          node('frog') {
            sh 'make PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=/sandbox/petsc/workspace/PETSc all'
          }
        }
        'n-gage' : {
          node('n-gage') {
            sh 'make PETSC_ARCH=arch-linux-pkgs-dbg-ftn-interfaces PETSC_DIR=/sandbox/petsc/workspace/PETSc all'
          }
        }
      )
    }
    
    stage('Examples') {
      node {
        sh "make -f gmakefile test search='tao_unconstrained%'"
      }
    }
  }
}
