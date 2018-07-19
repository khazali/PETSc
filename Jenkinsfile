#!/usr/bin/env groovy
def alljob = JOB_NAME.tokenize('/') as String[]
def node_name = alljob[0]
def arch_name = alljob[1]

pipeline { 
    agent { node { label node_name} }
    
    stages {
        stage('Local Merge') {
            steps {
                node {
                    label node_name
                    checkout scm
                }
            }
        }
        stage('Configure') {
            steps {
                node {
                    label node_name
                    sh "python ./config/examples/${arch_name}.py"
                }
            }
        }
        stage('Make') {
            steps {
                node {
                    label node_name
                    sh "make PETSC_ARCH=${arch_name} PETSC_DIR=${WORKSPACE} all"
                }
            }
        }
        stage('Examples') {
            steps {
                node {
                    label node_name
                    sh "make PETSC_ARCH=${arch_name} PETSC_DIR=${WORKSPACE} -f gmakefile test"   
                }
            }
        }
    }  
}
