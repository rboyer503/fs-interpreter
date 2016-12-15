# fs-interpreter
Fingerspelling Interpreter
==========================

Overview
--------
This software was developed as an exploratory effort to apply machine vision and machine learning algorithms to develop AI capable of interpreting a subset of sign language known as fingerspelling (https://en.wikipedia.org/wiki/Fingerspelling).  The software is designed to recognize the letters of the American Manual Alphabet (https://en.wikipedia.org/wiki/American_manual_alphabet) as well as a single "sentinel" symbol (open hand facing forward with fingers spread) used to indicate the end of a phrase.


Software usage is as follows:

1. Start the main FS Interpreter Python script.
2. Hold the 'A' symbol (basic closed hand facing forward with thumb on side) roughly 2 to 6 feet from webcam until software locates hand and enters the TRACK_HAND state.
3. Hold the sentinel symbol (open hand facing forward with fingers spread) to begin.  The FS Interpreter is now ready to begin accepting symbols.
4. Fingerspell a phrase using the American Manual Alphabet.
5. Hold the sentinel symbol to complete the phrase.  The FS Interpreter will then finalize analysis and read back the phrase it believes to be most likely.
6. Repeat inputting phrases as desired.


The software includes three major projects:

- FSI Data Collection Tool: A utility designed to simplify collection of data samples used for training the Convolutional Neural Nets (CNNs).
- Word Model Manager: A component used to enumerate and sort by probability "phrase candidates" based on an input sequence of character probability distributions.
- FS Interpreter: Core Python program to interpret fingerspelling.

See the corresponding README files in the subdirectories for details.


Limitations
-----------
This was largely an exploratory effort and various limitations exist with the software as it is currently implemented.  There was quite a bit of effort to improve the robustness of various aspects of the software (hand tracking, symbol classification, etc.)  Nevertheless, the software has certain known issues:

1. All training, validation, and test data sets exclusively include samples of my hand, with my own imperfect (and not necessarily representative) fingerspelling technique.  As a result, users may find that results are unimpressive, especially if the fingerspelling technique is considerably different from my own.  With a more expansive data collection effort, I believe it would be possible to improve the general performance.  In lieu of that, it is possible to tune the neural nets by performing some training on additional samples provided by the user.

2. Difficult lighting conditions (low light, harsh shadows, blown-out highlights, changes in lighting while running, etc.) and backgrounds including extensive skin tones can negatively impact the performance of the hand location and tracking algorithms (and in turn interfere with fingerspelling interpretation).  These algorithms are designed to be reasonably robust, but they have their limits.  If hand location or tracking is not performing well, consider changing the lighting or background.

3. The software is tuned to perform best when fingerspelling at a fairly slow rate (2 to 3 letters per second works best).  The software can be tuned to deal with faster fingerspelling, but unfortunately this comes at the cost of accuracy.  Some steps have been taken to attempt to achieve decent accuracy despite imperfect classification of symbols.  For example, the Word Model Manager takes as input the full probability distribution of each symbol and builds a model to assign probabilities to phrases limited to words found in a defined dictionary.  In addition, a more experimental Recurrent Neural Net (RNN) trained on a text corpus provides feedback on the likelihood of a given sequence of words for the most likely phrase candidates.  Nevertheless, this approach is not adequate to accurately interpret fast fingerspelling.

4. The Word Model Manager relies on a slightly modified SOWPODS dictionary (a dictionary used for English-language Scrabble tournaments).  The obvious shorter non-words have been removed.  Nevertheless, the dictionary is not ideal - it still includes some "words" that should probably be removed, and it doesn't include countless proper nouns, acronyms, etc.  Unfortunately, this means that any phrase that includes a proper noun or acronym which has not been explicitly added to the dictionary will not be interpreted correctly.  A further complication is that the RNN is trained on a limited text corpus which does not necessarily include proper nouns (or other words) that the user intends to use.  As a result, the RNN may under some circumstances work against the overall performance.  Further tuning of the dictionary and more expansive data collection for training the RNN would be needed to address these issues.

5. The hand tracking algorithm is not currently designed to adapt to changes in depth.  For optimal performance, keep your hand at a relatively constant distance from the webcam (3 feet works well).


Hand location, tracking, and feature extraction pipeline
--------------------------------------------------------
The following provides an outline of the stages of processing involved in finding and tracking the hand, and extracting the features used for training or live prediction.  This pipeline is implemented in both the FSI Data Collection Tool and the main FS Intepreter.

1. Stage 1: CAM_AUTO_ADJUST
  1. Brief delay while webcam performs auto-adjust (for exposure, brightness, contrast, white-balance, etc.)
  2. After delay, transition to FIND_HAND stage.
2. Stage 2: FIND_HAND
  1. Convert to grayscale image and equalize histogram.
  2. Run multi-scale detection using cascade classifier to find 'A' symbol.
  3. Tighten requirements if more than one hand is detected (i.e.: increment "minimum neighbors").
  4. If one and only one hand is detected, proceed.
  5. Establish ROI (Region of Interest) and initialize a loose bounding box to track hand location.
  6. Filter out false positives by calculating an "outlier factor" based on image statistics.  See ImageProcessCoordinator::GetHandOutlierFactor() in the FSI Data Collection Tool for details.
  7. Gaussian blur on ROI, 5x5 kernel.
  8. Initialize "focus weights", a matrix built using parabolic equations - later used to give higher priority to pixels located closer to the center of the ROI.
  9. Build LUT (Lookup Table); entries with large positive values correspond to YCrCb channel values commonly found in the ROI, but not the full image.  See ImageProcessCoordinator::BuildLUT() in the FSI Data Collection Tool for details.
  10. Transition to the TRACK_HAND stage.
3. Stage 3: TRACK_HAND
  1. Establish ROI on loose bounding box.
  2. Gaussian blur on ROI, 5x5 kernel.
  3. Convert to YCrCb color space.
  4. Apply LUT to generate color-based quality matrix to give priority to pixels with hand colors.
  5. Apply custom adaptive Canny edge detection algorithm to build edge map.  See the AdaptiveCanny class in the FSI Data Collection Tool for details.
  6. Apply distance transform algorithm on edge map and translate raw distance-to-edge data into weights using parabolic equation to build edge-based quality matrix.  Used to give higher priority to pixels located close to an edge.
  7. Normalize and multiply the color-based quality matrix, the edge-based quality matrix, and the focus weights matrix to build a master quality matrix.
  8. Calculate the "center of mass" of the master quality matrix.
  9. Adjust the loose bounding box to center on the calculated centroid.

Tracking is thus performed by continually adjusting the position of the loose bounding box around the perceived location of the hand.  The master quality matrix serves a dual role.  It's used for hand tracking, and it's also the actual feature matrix used for training and live predictions.