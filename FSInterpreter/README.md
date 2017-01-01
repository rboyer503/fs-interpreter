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


Requirements/Dependencies
-------------------------
- 64-bit Linux OS
  - I use Ubuntu 14.04 LTS.
  - 32-bit support may be possible, but there may be some complications with TensorFlow and its build tool, Bazel.
- Python 2.7
  - Adapting to Python 3.3 should be straight-forward to the best of my knowledge, but this is my first foray into Python scripting.
- TensorFlow 0.8
  - Exclusively using Python API.
  - No testing has yet been completed on newer releases.
- OpenCV 2.4.11 (Core, HighGUI, ImgProc, and ObjDetect)

The Fingerspelling Interpreter utilizes many computationally expensive machine vision and learning algorithms that can greatly benefit from GPU acceleration.  To achieve the optimal frame rate while running FSInterpreter.py and to reduce CNN training session times, it is strongly recommended to leverage CUDA and the cuDNN on a machine with a suitable GPU and a 64-bit Linux OS.  I use an ASUS G751JY laptop with an NVIDIA GeForce GTX 980M (Maxwell architecture) with the following additional dependencies:
- CUDA 7.5
- cuDNN 4.0


FSInterpreter.py
----------------
This Python script implements the core hand tracking and fingerspelling interpretation.

### Usage
    python FSInterpreter.py <device ID>
        <device ID>: ID of video device (0-9)

The full hand location, tracking, and feature extraction pipeline is implemented, as detailed in the top-level README.  After initialization, there is a brief delay to allow the webcam to perform its auto-adjust.  Then the software will begin searching for your hand.

The current state is displayed at the top of the main window.  Once "Finding hand..." is displayed, hold the "A" symbol (basic closed hand facing forward with thumb on side) roughly 2 to 6 feet from webcam.  Once the hand is located, the software will run a custom exposure tuning process that takes several seconds.  Once the tuning process is complete, the software will transition to the main tracking state and a bounding box will be continually adjusted to track the location of your hand.

Once "Tracking hand..." is displayed, a diagnostic display will reflect the output of the smoothed, merged CNN predictions for the current symbol; letters with a probability exceeding a minimum threshold will be rendered as capital letters with the font size scaled by the probability.  So, for example, when you make the "B" symbol, the diagnostic display should show a large "B" character and possibly some other very small characters resulting from imperfect classification.

No phrase interpretation will begin until the first sentinel symbol.  To begin hold the sentinel symbol (open hand facing forward with fingers spread).  Then fingerspell a short phrase, ending with the sentinel symbol.  The second diagnostic display (just above the aforementioned one) will report the its most probable phrase candidate on-the-fly, finalizing the prediction once the sentinel symbol is received.
