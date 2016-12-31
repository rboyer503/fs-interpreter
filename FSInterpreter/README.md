Fingerspelling Interpreter
==========================

Overview
--------
This is the core Fingerspelling Interpreter, consisting of several Python scripts which utilize machine vision algorithms and machine learning models to interpret fingerspelling (specifically the American Manual Alphabet).

The directory structure is as follows:
- fs-interpreter/FSInterpreter
  - data
    - This directory contains the raw, gzipped data and label files generated from the FSI Data Collection Tool as well as the SOWPODS dictionary used by the Word Model Manager for word validation.
  - lib
    - This directory contains the Word Model Manager shared library, libwordmodelmgr.so.
  - model
    - jz-cnn
      - logs
        - This directory contains backed up log files from training sessions of the J/Z CNN model.
      - This directory contains the official model parameters for the J/Z CNN model (train-vars-best-jz) and also houses temporary files and logs during J/Z CNN model training.
    - main-cnn
      - logs
        - This directory contains backed up log files from training sessions of the main CNN model.
      - This directory contains the official model parameters for the main CNN model (train-vars-best-main) and also houses temporary files and logs during main CNN model training.
    - This directory contains the .xml classifier file for the closed fist Haar classifier used by the cascade classifier for initial hand location (aHand.xml).
  - sounds
    - This directory contains any sound files (beep.wav).
  - The top-level directory contains the core Python and shell scripts supporting model training and fingerspelling interpretation, including:
    - FSInterpreter.py - The core script for hand tracking and fingerspelling interpretation.
    - TrainJzCNN.py - The script for running training sessions to optimize model parameters for the J/Z CNN model.
    - TrainMainCNN.py - The script for running training sessions to optimize model parameters for the main CNN model.
    - wmmWrapper.py - Provides Python wrapper class, WordModelMgr, for interfacing with the Word Model Manager shared library.
    - do_train_jz_multi.sh - Shell script to help automate hyper-parameter optimization of the J/Z CNN model.
    - do_train_main.sh - Shell script to loop through several training sessions of the main CNN model.
    - do_train_main_multi.sh - Shell script to help automate hyper-parameter optimization of the main CNN model.
    - write_backup.sh - Shell script to help automate backup and summarization of log files during training of the main CNN model.
    - write_backup_jz.sh - Shell script to help automate backup and summarization of log files during training of the J/Z CNN model.


