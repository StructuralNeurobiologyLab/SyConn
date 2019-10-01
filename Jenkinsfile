pipeline {
  agent any
  environment {
      PATH="/u/pschuber//anaconda3/bin:$PATH"
    }
  stages {
    stage('build') {
      steps {
        sh '''conda create --yes -n pysyintegration python
            conda activate pysyintegration
            pip install -r requirements.txt
            pip install -e .
            '''
      }
    }
    stage('test') {
      steps {
          sh '''
            conda activate pysyintegration
            python -m pytest --junit-xml=pytest_unit.xml
            '''
      }
    }
  }
  post {
    always {
        sh 'conda remove --yes -n pysyintegration --all'
    }
    failure {
        echo "Error while removing conda environment."
    }
  }
}