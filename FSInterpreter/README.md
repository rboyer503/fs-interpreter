Fingerspelling Interpreter
==========================

Overview
--------
This is the core Fingerspelling Interpreter, consisting of several Python scripts which utilize machine vision algorithms and machine learning models to interpret fingerspelling (specifically the American Manual Alphabet).

The directory structure is as follows:
- fs-interpreter/FSInterpreter
  - data
    - This directory contains various files including:
      - The raw, gzipped data and label files generated from the FSI Data Collection Tool
      - The SOWPODS dictionary used by the Word Model Manager for word validation.
      - The training, validation, and test data files from the Penn Treebank corpus for the Phrase Model Manager.
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
    - pmm-rnn
      - This directory contains the official model parameters for the LSTM RNN model used for the Phrase Model Manager.
    - This directory contains the .xml classifier file for the closed fist Haar classifier used by the cascade classifier for initial hand location (aHand.xml).
  - sounds
    - This directory contains any sound files (beep.wav).
  - The top-level directory contains the core Python and shell scripts supporting model training and fingerspelling interpretation, including:
    - FSInterpreter.py - The core script for hand tracking and fingerspelling interpretation.
    - TrainJzCNN.py - The script for running training sessions to optimize model parameters for the J/Z CNN model.
    - TrainMainCNN.py - The script for running training sessions to optimize model parameters for the main CNN model.
    - TrainPMMRNN.py - The script for running training sessions to optimize model parameters for the Phrase Model Manager's LSTM RNN.
    - phraseModelMgr.py - Provides Python class, PhraseModelMgr, for evaluating the perplexity of proposed phrases.
    - pmmTest.py - A simple test harness for the Phrase Model Manager.
    - reader.py - Provides utility functions for parsing Penn Treebank text files.
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
  - Adapting to Python 3.3 should be straight-forward to the best of my knowledge.
- TensorFlow 0.8
  - Exclusively using Python API.
  - No testing has yet been completed on newer releases.
- OpenCV 2.4.11 (Core, HighGUI, ImgProc, and ObjDetect)

The Fingerspelling Interpreter utilizes many computationally expensive machine vision and learning algorithms that can greatly benefit from GPU acceleration.  To achieve the optimal frame rate while running FSInterpreter.py and to reduce CNN/RNN training session times, it is strongly recommended to leverage CUDA and the cuDNN on a machine with a suitable GPU and a 64-bit Linux OS.  I use an ASUS G751JY laptop with an NVIDIA GeForce GTX 980M (Maxwell architecture) with the following additional dependencies:
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

No phrase interpretation will begin until the first sentinel symbol.  To begin, hold the sentinel symbol (open hand facing forward with fingers spread).  Then fingerspell a short phrase, ending with the sentinel symbol.  The lower part of the diagnostic display will report the output of the Word Model Manager and Phrase Model Manager (prefixed by "WMM:" and "PMM:" respectively).  The Word Model Manager output is reported on-the-fly, finalizing the prediction once the sentinel symbol is received.  The final sentinel also triggers the Phrase Model Manager processing; a small number of the best phrases returned by the Word Model Manager are reevaluated based on their perplexity.

### Commands
The following commands are available via key press while the main display window is active.

    ESC:    Exit
    1:      Dump all phrase candidates from Word Model Manager
    2:      Dump full phrase info from Phrase Model Manager
    [a..z]: Capture a burst of data samples to data.raw/labels.raw
    +/-:    Increase/decrease camera exposure

### CNN Tuning Support
In order to close the loop between identifying classification failures and generating new data samples to improve classification accuracy, functionality is available to capture samples on the fly.  While a symbol is being misclassified, hit the corresponding key to append a sample to local raw data files, data.raw and labels.raw.  These supplemental samples can then be used to tune the main CNN.

Tuning the main CNN involves gzipping the data.raw and labels.raw files, then running one or more training sessions using the TrainMainCNN.py script with the --tune option.  This will overwrite random training samples from the standard data files with the supplemental tuning data.  Additionally, the test error against the tuning samples is periodically reported so that improvement can be monitored.  Once you're satisfied with the results, replace the ./model/main-cnn/train-vars-best-main checkpoint with the newly generated ./model/main-cnn/train-vars-best.

Note that tuning of the J/Z CNN is not currently supported.


Convolutional Neural Network Models
-----------------------------------
The fingerspelling interpreter uses two separate CNN models, both typical CNN architectures.  The "main" CNN model is used for interpretation of all letters that don't involve motion, including the sentinel symbol.  The "J/Z" CNN model covers the remaining two letters, J and Z, which involve motion.  The models have the same essential architecture as outlined below.

### Main CNN Layers
- Conv layer 1
  - Convolution with 5x5 kernel, ReLU activation, and 2x2 max pooling.
- Conv layer 2
  - Convolution with 5x5 kernel, ReLU activation, and 2x2 max pooling.
- Conv layer 3
  - Convolution with 5x5 kernel, ReLU activation, and 2x2 max pooling.
- Conv layer 4
  - Convolution with 5x5 kernel, ReLU activation, and 2x2 max pooling.
- Fully-connected layer 1
  - Fully-connected, ReLU activation.
- Fully-connected layer 2
  - Fully-connected, ReLU activation.
- Softmax layer
  - Transformation to produce logits.

### J/Z CNN Layers
Same as above, except using 3x3 kernels.

### CNN Model Inputs
Input to the models is the "master quality matrix" calculated as part of the hand location, tracking, and feature extraction pipeline described in the top-level README.  This master quality matrix is a small, grayscale image with pixel intensity proportional to the estimated probability that the pixel is part of the hand.  For the main CNN, the input to the model is a single 80x80 matrix.  For the J/Z CNN, the input to the model is an array of 8 64x64 matrices providing a short video clip of the J or Z gesture.

For training, the input matrices are extracted from gzipped files containing the raw data.  The raw files can be generated using the FSI Data Collection Tool.  Data and corresponding labels are stored in separate files; for example, the first set of data and labels for the main CNN are stored in the files `data1.raw.gz` and `labels1.raw.gz` respectively.  Separate data and label files exist for the test sets, e.g.: `data-test.raw.gz` and `labels-test.raw.gz`.

Note that no data multiplication techniques are used.  Data multiplication is handled as part of the data collection process.  (The FSI Data Collection Tool generates additional data samples by applying some basic distortions such as random rescaling, rotation, contrast adjustment, and cropping.)

Input samples are converted to floating point, scaled to the range -0.5 to 0.5.

### CNN Model Training
The models are trained using mini-batch gradient descent applying the momentum optimization algorithm with a 0.9 momentum term.  The learning rate decays once per epoch, following an exponential schedule.  The objective function for the models is the sum of the cross entropy loss combined with weight decay (L2 regularization) of the parameters for the fully-connected layers.  During training, dropout is also applied.  Data is reshuffled after each epoch.  I experimented with batch normalization as well, but this was eventually removed from the model since the expected improvements never materialized.

Note that the current implementation is not following best practices with respect to proper validation and testing.  The held-out test data has been extensively used for tuning of hyperparameters and the "validation data" is simply a sample of the training data which is randomly re-selected for each epoch.  Proper segregation of training, validation, and test samples should be implemented going forward.

### Hyperparameter Optimization
Quite a bit of time was applied towards optimizing hyperparameters.  The approach taken was a simple grid search.  Separate "grids" were searched separately, each investigating a handful of tightly related parameters to mitigate the "curse of dimensionality".  Some fundamental hyperparameters and the core model architecture were determined prior to some cleanup activity, so the training scripts parameterize only a subset of the hyperparameters.  These include:
- Base convolutional depth (depth of the first convolutional layer)
  - For each subsequent conv layer, the depth doubles.
- Fully-connected layer size (number of neurons between the first and second fully-connected layer)
- Initial learning rate
- Learning rate decay term
- Keep probability (for dropout)
- L2 regularization factor
- Mini-batch size


Phrase Model Manager
--------------------
This component is responsible for evaluating the relative perplexity of phrases returned by the Word Model Manager.  The intention of the Phrase Model Manager is to improve the overall robustness of the fingerspelling interpretation by providing feedback from a higher level of abstraction.  A Recurrent Neural Network LSTM model trained on data from the Penn Treebank corpus is used to evaluate perplexity, a heuristic estimating how unlikely a given sequence of words is.  As an example, suppose the Word Model Manager returns the following two phrase candidates:
- MY NAME IS ROB
- MYNA ME IS ROB

Although both phrases can be made with the same sequence of letters, the first phrase is a very common statement, while the second phrase uses a very uncommon word and isn't even grammatically correct.  The Phrase Model Manager aims to incorporate this information.

Note that this feature is experimental, even by the standards of this project.

### RNN LSTM Model
Only minor changes are made to the Penn Treebank LSTM model example included with Google's TensorFlow.  This is not ideal for this application, but is convenient and workable for testing an experimental feature.
