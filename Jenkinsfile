pipeline {
  agent {
    node {
      label 'frog'
    }
  }
  
  stages {
    stage('Configure') {
      steps {
        sh './configure --download-suitesparse --download-mumps --download-scalapack --download-chaco -
        -download-ctetgen --download-exodusii --download-cmake --download-pnetcdf --download-generator --download-hdf5 --download-z
        lib=1 --download-metis --download-ml --download-netcdf --download-parmetis --download-triangle --with-cuda --with-shared-li
        braries PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=/sandbox/petsc/petsc.next-3'
      }
    }
    stage('Build') {
      sh 'make PETSC_ARCH=arch-c-exodus-dbg-builder PETSC_DIR=/sandbox/petsc/petsc.next-3 all'
    }
    stage('Test') {
      sh "make -f gmakefile test search='tao%'"
    }
  }
}
