#!/usr/bin/env groovy

pipeline {
    environment{
        NODE_NAME=$(basename $(dirname ${JOB_NAME}))
    }
    
    agent env.NODE_NAME
    
    stages {
        stage('Configure') {
            steps {
                checkout scm
                sh 'python ./config/examples/${JOB_BASE_NAME}.py'
            }
        }
        stage('Make') {
            steps {
                sh 'make PETSC_ARCH=${JOB_BASE_NAME} PETSC_DIR=${WORKSPACE} all'
            }
        }
        stage('Examples') {
            steps {
                sh 'make PETSC_ARCH=${JOB_BASE_NAME} PETSC_DIR=${WORKSPACE} -f gmakefile test'    
            }
        }
    }
}
