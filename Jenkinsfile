pipeline {
  agent { docker { image 'python:3.7.2' } }
  stages {
    stage('build') {
      steps {
        sh 'python setup.py develop'
      }
    }
    stage('test') {
      steps {
        sh 'pytest'
      }
    }
  }
}