#!/usr/bin/env groovy
def alljob = JOB_NAME.tokenize('/') as String[]
def node_name = alljob[0]
echo node_name
echo "${node_name}"
def arch_name = alljob[1]
echo arch_name
echo "${arch_name}"

pipeline { 
    agent { node { label node_name} }
    
    stages {
        stage('Local Merge') {
            steps {
                checkout scm
            }
        }
        stage('Configure') {
            steps {
                checkout scm
                sh 'python ./config/examples/${arch_name}.py'
            }
        }
        stage('Make') {
            steps {
                sh 'make PETSC_ARCH=${arch_name} PETSC_DIR=${WORKSPACE} all'
            }
        }
        stage('Examples') {
            steps {
                sh 'make PETSC_ARCH=${arch_name} PETSC_DIR=${WORKSPACE} -f gmakefile test'    
            }
        }
    }  
}
