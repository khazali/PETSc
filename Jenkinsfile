#!/usr/bin/env groovy
def alljob = JOB_NAME.tokenize('/') as String[]
def node_name = alljob[0]

pipeline { 
    agent { node { label node_name} }
    
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
