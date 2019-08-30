pipeline {
  agent any

  environment {
      PATH="/u/pschuber//anaconda3/bin:$PATH"
    }

  stages {
    stage('build') {
      steps {
        sh '''
            echo $PATH
            which python
            python -m pytest --junit-xml=pytest_unit.xml
            echo $PATH
            which pip
            pip install -r docs/requirements.txt
            /u/pschuber/anaconda3/bin/pip install -r docs/requirements.txt

            ls /u/pschuber/anaconda3/bin/python
            which conda
            which sh
            conda activate pysy
            /u/pschuber/anaconda3/bin/pip install -r docs/requirements.txt
            pip install -e .
            '''
      }
    }
    stage('test') {
      steps {
            sh '''conda create --yes -n pysyintegration python
                source activate pysyintegration
                pip install -r requirements.txt
                '''
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
        echo "Send e-mail, when failed"
    }
  }
}