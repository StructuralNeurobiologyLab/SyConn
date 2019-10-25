pipeline {
  agent any
  environment {
      PATH="/wholebrain/u/atultm/anaconda3/bin:$PATH"
    }
  stages {
    stage("build") {
      steps {
            sh """
                echo $USER
                ls /wholebrain/u/atultm/anaconda3/bin/python
                which conda
                which sh
                which pip
            """
            sh 'printenv | sort'

            sh """#!/bin/bash
                /wholebrain/u/atultm/anaconda3/bin/python -c 'print("hello")'
                /wholebrain/u/atultm/anaconda3/bin/pip install -r docs/requirements.txt
                /wholebrain/u/atultm/anaconda3/bin/pip install -e .
                source /wholebrain/u/atultm/anaconda3/etc/profile.d/conda.sh
                conda activate qaz

            """
      }
    }
    stage('test') {
      steps {
          sh """#!/bin/bash
            source /wholebrain/u/atultm/anaconda3/etc/profile.d/conda.sh
            conda activate qaz
            python -m pytest --junit-xml=pytest_unit.xml
            """
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
