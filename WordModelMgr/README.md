Word Model Manager
==================

Overview
--------
This component is responsible for managing a set of all possible phrases intended by the user (referred to as "phrase candidates") based on the predictions from the CNNs mediated by the FS Interpreter's State Manager.  As letter predictions are added by the State Manager, the Word Model Manager component builds new phrase candidates that are consistent with the loaded dictionary, updates and normalizes the relative probabilities, and culls all phrase candidates that fall below a minimum threshold.

The Word Model Manager improves the overall robustness of the fingerspelling interpretation by providing feedback from the higher-level domain of words.  Although our letter predictions are imperfect, the majority of incorrect predictions will result in phrases containing non-words.

This component was written in C++ and the makefile builds a shared library suitable for integration into the Fingerspelling Interpreter.  A wrapper is provided to adapt the Word Model Manager library to a Python class, WordModelMgr.

The directory structure is as follows:
- fs-interpreter/WordModelMgr
  - data
    - This directory contains the SOWPODS dictionary used for word validation.
  - lib
    - This is the output directory for the shared library.
  - obj
    - Temporary directory structure for debug and release object code.
  - src
    - This directory contains the source code files for the Word Model Manager shared library.
  - The top-level directory contains:
    - Makefile - Makefile for building shared library, libwordmodelmgr.so.
    - wmmWrapper.py - The ctypes wrapper providing the WordModelMgr Python class.
    - wmmTest.py - A simple test harness used during development.


Building
--------
To build the shared library:
    make release
or
    make debug

The shared library will be placed in fs-interpreter/WordModelMgr/lib.  This must be copied to fs-interpreter/FSInterpreter/lib for use by FSInterpreter.py.


Dependencies
------------
- Boost 1.55 (Only for posix_time - could be removed without much effort.)
- Python 2.7
  - Adapting to Python 3.3 should be straight-forward to the best of my knowledge.
- GNU GCC/make


WordModelMgr class interface
----------------------------
### __init__(self)
- Construct Word Model Manager instance.

### __del__(self)
- Free instance.

### initialize(self)
- Load dictionary and prepare baseline state.

### add_letter_prediction(self, letter_index, confidence, double_prob)
   letter_index - 0="A", ..., 25="Z", 26=sentinel
   confidence - the confidence in the prediction from 0.0 to 1.0
   double_prob - probability that letter is a double (e.g.: the oo in too)
- Build new phrase candidates based on specified letter prediction.

### finalize_prediction(self)
- Remove phrase candidates that end in the middle of a word.

### get_best_prediction(self)
- Retrieve the text of the phrase candidate with highest relative probability.

### get_next_prediction(self, prob)
    prob - output argument; will supply actual relative probability of returned phrase
- First call after `add_letter_prediction` or `finalize_prediction` returns the text of the phrase candidate with highest relative probability.
- Subsequent calls return next best phrase candidate.

### reset(self)
- Reset to baseline state to prepare for new phrase.

### dump_candidates(self)
- Sort and print all phrase candidates and their relative probabilities for diagnostic purposes.

