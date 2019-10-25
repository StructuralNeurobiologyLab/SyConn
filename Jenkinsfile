pipeline {
  agent any
  environment {
      PATH="/wholebrain/scratch/CI/atulconda1/bin:$PATH"
    }
  stages {
    stage("build") {
      steps {
            sh """
                echo $USER
                ls /wholebrain/scratch/CI/atulconda1/bin/python
                which conda
                which sh
                which pip
            """
            sh 'printenv | sort'

            sh """#!/bin/bash
                /wholebrain/scratch/CI/atulconda1/bin/python -c 'print("hello")'
                /wholebrain/scratch/CI/atulconda1/bin/pip install -r docs/requirements.txt
                /wholebrain/scratch/CI/atulconda1/bin/pip install -e .
                source /wholebrain/scratch/CI/atulconda1/etc/profile.d/conda.sh
                conda activate qazz

            """
      }
    }
    stage('test') {
      steps {
          sh """#!/bin/bash
            source /wholebrain/scratch/CI/atulconda1/etc/profile.d/conda.sh
            conda activate /wholebrain/scratch/CI/atulconda1/envs/qazz
            python -m pytest --junit-xml=pytest_unit.xml
            """
      }
    }
  }
  post {
    always {
        sh 'conda remove --yes -n qazz --all'
    }
    failure {
        echo "Error while removing conda environment."
    }
  }
}
