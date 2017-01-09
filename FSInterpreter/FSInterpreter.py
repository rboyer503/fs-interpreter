"""
Fingerspelling Interpreter
Perform hand tracking and fingerspelling interpretation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import time
import math
import os
import argparse

from wmmWrapper import *
from phraseModelMgr import *


# ================================================================================
# Constants
# ================================================================================
# Miscellaneous constants:
PIXEL_DEPTH = 255
SEED = None  # 66478
PROFILE_ACTIVE = False
MAX_TUNING_FRAMES = 150
BLOWOUT_TEST_FREQ = 10
BLOWOUT_THRES = 100
PREDICT_HIST_SIZE = 8
NUM_SYMBOLS = 27
NUM_TOP_PHRASES = 5

# Main CNN-specific constants:
# Update if CNN hyper-parameters or data sets are modified.
IMAGE_SIZE = 80
NUM_IMAGES = 142500
NUM_LABELS = 25
VALIDATION_SIZE = 1200
TEST_SIZE = 3192
KERNEL_SIZE = 5
CONV_DEPTH = 10
FC_DEPTH = 100
LR_INIT = 0.015
LR_DECAY = 0.95
KEEP_PROB = 0.5
REG_FACTOR = 0.0005
BATCH_SIZE = 100

# J/Z CNN-specific constants:
# Update if CNN hyper-parameters or data sets are modified.
CLIP_IMAGE_SIZE = 64
CLIP_NUM_CHANNELS = 8
CLIP_NUM_IMAGES = 7200
CLIP_NUM_LABELS = 3
CLIP_VALIDATION_SIZE = 100
CLIP_TEST_SIZE = 1440
CLIP_KERNEL_SIZE = 3
CLIP_CONV_DEPTH = 5
CLIP_FC_DEPTH = 220
CLIP_LR_INIT = 0.013
CLIP_LR_DECAY = 0.99
CLIP_KEEP_PROB = 0.5
CLIP_REG_FACTOR = 0.0005
CLIP_BATCH_SIZE = 75

# Hand ROI statistics constants:
AVG_MEAN = (110.491 * 1.2, 150.943, 109.697)
AVG_STD_DEV = (36.3768 * 0.8, 7.31453, 8.04819)
MAX_OUTLIER_FACTOR = 6.0

# Bounding box constants:
BOUND_LOOSE_FACTOR = 2.5
BOUND_LOOSE_OFFSET = ((BOUND_LOOSE_FACTOR - 1.0) * 0.5)
BOUND_TIGHT_FACTOR = 0.35
BOUND_TIGHT_OFFSET = ((1.0 - BOUND_TIGHT_FACTOR) * 0.5)
BOUND_FOCUS_FACTOR = 0.9
BOUND_FOCUS_OFFSET = ((1.0 - BOUND_FOCUS_FACTOR) * 0.5)
TIGHT_AREA = (BOUND_TIGHT_FACTOR * BOUND_TIGHT_FACTOR)


# ================================================================================
# Enumerations
# ================================================================================
class FSIMode:
    """
    Fingerspelling Interpreter Mode enumeration.
    """
    FSI_INIT = 0    # Initial state
    FSI_DELAY = 1   # Delay for delayDur seconds
    FSI_FIND = 2    # Locate hand in image
    FSI_TUNE = 3    # Hysteresis phase in which the main CNN is used to reject false positives and exposure is tuned
    FSI_TRACK = 4   # Main phase in which hand is tracked and fingerspelling is interpreted

    def __init__(self):
        pass


class HandAcceptance:
    """
    Hand Acceptance enumeration.
    """
    HA_PROCESSING = 0   # Exposure tuning is on-going
    HA_REJECT = 1       # Hand candidate has been rejected
    HA_ACCEPT = 2       # Hand candidate has been accepted

    def __init__(self):
        pass


# ================================================================================
# Globals
# ================================================================================
pdf = np.empty((PREDICT_HIST_SIZE, 1))
profileTime = time.time()
wmm = None
pmm = None
mainGraph = None
mainSess = None
jzGraph = None
jzSess = None
devName = None
stateMgr = None
currMode = FSIMode.FSI_INIT
fsiModeString = ["Initializing...",
                 "Waiting for webcam auto-adjust...",
                 "Locating hand...",
                 "Tuning exposure...",
                 "Tracking hand..."]
delayDur = 5
startTime = time.time()
tuneModeRemaining = MAX_TUNING_FRAMES
tuneNext = BLOWOUT_TEST_FREQ
tuneLastBlowout = 0
tuneComplete = False
anchorSize = 0
sizeAdjustArray = np.zeros(10, dtype=np.int)
sizeAdjustIndex = 0
absoluteExposure = 400
cascade = cv2.CascadeClassifier('model/aHand.xml')
minNeighbors = 3
looseBound = None
focusWeights = None
lut = None
adaptiveCanny = None
imgFeatureMain = None
imgFeatureJZ = None
numPredictions = 0
lastNumPredictions = 0
lastTime = time.time()
clipArray = np.zeros((1, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE, CLIP_NUM_CHANNELS), dtype=np.float32)
rollingPredictions = np.zeros((PREDICT_HIST_SIZE, NUM_SYMBOLS), dtype=np.float32)


# ================================================================================
# Classes
# ================================================================================
class AdaptiveCanny:
    """
    This class implements a simple adaptive Canny edge detector.
    It uses OpenCV's Canny() function for the core Canny algorithm, but adjusts the thresholds dynamically to keep a
    relatively constant proportion of edge pixels within a circular area most likely to contain the hand.
    """
    EDGE_FACTOR_EPSILON = 0.00125
    INIT_HIGH_THRES = 100.0
    MIN_HIGH_THRES = 30.0
    MAX_STEPS = 100

    def __init__(self, edge_factor, step_size, gradual):
        """
        Initialize instance.
        :param edge_factor: fraction of pixels that should be edges
        :param step_size: value to adjust high threshold per step
        :param gradual: boolean, if true only one pass will be performed per process call, otherwise runs several passes
                        to attempt full convergence to optimal thresholds in one call
        """
        self._low_edge_factor = edge_factor * (1 - AdaptiveCanny.EDGE_FACTOR_EPSILON)
        self._high_edge_factor = edge_factor * (1 + AdaptiveCanny.EDGE_FACTOR_EPSILON)
        self._step_size = step_size
        self._gradual = gradual
        self._high_thres = AdaptiveCanny.INIT_HIGH_THRES
        self._low_thres = AdaptiveCanny.INIT_HIGH_THRES / 2

    def process(self, image):
        """
        Process grayscale image to generate its Canny edge map, adapting thresholds dynamically.
        :param image: grayscale hand image ROI (loose bounding box)
        :return: success: boolean, if true, either gradual mode was selected or desired edge count was achieved
                 edges: Canny edge map
        """
        # Calculate edge count thresholds.
        height, width = image.shape[:2]
        low_edge_count = height * width * self._low_edge_factor
        high_edge_count = height * width * self._high_edge_factor

        # Run adaptive algorithm to generate canny mask.
        # If gradual mode selected, only perform one pass and always return true.
        # Otherwise, perform several passes and return false if threshold adjustment never acheives desired edge count.
        edge_count_dir = 0
        edges = None
        if self._gradual:
            remaining_steps = 1
        else:
            remaining_steps = AdaptiveCanny.MAX_STEPS
        while remaining_steps > 0:
            # Perform edge detection with current thresholds.
            edges = cv2.Canny(image, self._low_thres, self._high_thres)

            # Get total number of edge pixels and a dispersion factor, reflecting the average distance from the center.
            edge_count, dispersion = self.calc_edge_stats(edges)

            # Adjust edge count to heuristic estimate of edges within tight circular boundary (i.e.: edges most likely
            # to be the hand).
            edge_count = int(edge_count * TIGHT_AREA)
            if dispersion > 0.0:
                edge_count /= (dispersion * math.pi)

            # Adjust Canny threshold parameters by step size if edge count is not in desired range.
            # Note: Imposing hard lower limit on high threshold to avoid excessive noise.
            if edge_count < low_edge_count:
                if (edge_count_dir == 1) or (self._high_thres < AdaptiveCanny.MIN_HIGH_THRES):
                    break
                else:
                    edge_count_dir = -1
                self._high_thres -= self._step_size
                self._low_thres = self._high_thres / 2.0
            elif edge_count > high_edge_count:
                if edge_count_dir == -1:
                    break
                else:
                    edge_count_dir = 1
                self._high_thres += self._step_size
                self._low_thres = self._high_thres / 2.0
            else:
                break
            remaining_steps -= 1
        return (self._gradual or (remaining_steps > 0)), edges

    @staticmethod
    def calc_edge_stats(edges):
        """
        Calculate number of edge pixels as well as a "dispersion factor" which reflects how far edge pixels typically
        are from the center.
        Note: It's possible this can be done more efficiently.
        :param edges: edge map from Canny
        :return: edge count and dispersion factor
        """
        edge_count = 1
        total_dist = 0
        height, width = edges.shape[:2]
        x_center = width // 2
        y_center = height // 2
        for x in xrange(width):
            for y in xrange(height):
                if edges.item(y, x) != 0:
                    edge_count += 1
                    total_dist += (x - x_center) ** 2
                    total_dist += (y - y_center) ** 2
        dispersion = total_dist / float(edge_count * width * height)
        return edge_count, dispersion


class StateMgr:
    """
    This class is responsible for maintaining state information used to bridge the gap between the raw predictions and
    the actual triggered letters forwarded to the Word Model Manager.
    Every frame, the smoothed, merged predictions are processed.  Each letter (plus the sentinel) is handled
    independently.
    When our confidence in a letter exceeds its current threshold, this moves that letter into a hysteresis phase in
    which we wait for some maximum number of frames for that letter's confidence to maximize.  This provides time for
    the user to complete proper formation of the letter.  The maximum confidence is then forwarded to the Word Model
    Manager.
    Each triggered letter also causes the letter's threshold to increase to a maximum value.  Once there is some
    indication the user is transitioning away from the letter (as detected by change in loss), the threshold follows a
    simple linear schedule to return to the default value.  This helps avoid duplicate triggers.
    Note that the confidence in a letter is scaled by (1.0 - loss) to factor in "noise".
    We also keep track of the X position of the loose bounding box so we can provide an estimated probability that the 
    user intended a double letter.
    Many of the "hyper-parameters" involved in the State Manager and Word Model Manager are currently tuned informally
    based on experimentation.  A more formal treatment, perhaps involving labeled video clips and an evolutionary
    algorithm, would improve accuracy.
    """
    SENTINEL_INDEX = 26
    SENTINEL_THRES = 12  # Number of consecutive sentinel detections to trigger finalization.
    DEF_THRES = 0.5  # Baseline threshold for triggering letter hysteresis.
    MAX_THRES = 1.5  # Maximum threshold to reduce sensitivity for a duration after letter triggered.
    THRES_STEP = 1.0 / 10  # Threshold step to reduce per frame while restoring baseline threshold.
    HYSTERESIS_DUR = 10  # Duration of hysteresis phase in frames.
    LOSS_THRES = 0.001  # Loss delta threshold; loss difference exceeding threshold indicates transition.

    # Constants controlling calculation of probability that letter is doubled, e.g.: to vs. too.
    # Thresholds are fraction of loose bounding box width.  The X position of the bounding box is tracked and these
    # thresholds are used to convert that "x-diff" to a probability in the range [0.1 .. 0.9].
    DOUBLE_PROB_MIN_THRES = 0.05
    DOUBLE_PROB_MAX_THRES = 0.20
    DOUBLE_PROB_THRES_RANGE = DOUBLE_PROB_MAX_THRES - DOUBLE_PROB_MIN_THRES

    class State:
        """
        Processing state enumeration.
        """
        PS_INIT = 0         # First sentinel not received
        PS_PROCESSING = 1   # Normal processing state
        PS_WAITING = 2      # Last phrase finalized, waiting for first new letter of next phrase

        def __init__(self):
            pass

    def __init__(self):
        """
        Initialize instance.
        """
        self._sentinel_count = 0
        self._thres = [StateMgr.DEF_THRES] * NUM_SYMBOLS
        self._hysteresis = [0] * NUM_SYMBOLS
        self._max_conf = [0.0] * NUM_SYMBOLS
        self._x_hist = [0] * StateMgr.HYSTERESIS_DUR
        self._x_hist_index = 0
        self._state = StateMgr.State.PS_INIT
        self._last_time = time.time()
        self._loss = [0.0] * NUM_SYMBOLS
        self._suspend = [False] * NUM_SYMBOLS

    def process_predictions(self, predictions):
        """
        Process latest predictions to update state and interface with the Word Model Manager.
        :param predictions: smoothed, merged predictions from the CNNs
        """
        global wmm
        global pmm

        # Update X-pos history.
        self.update_x_hist()

        # Debounce processing for sentinel symbol.
        sentinel_active = False
        max_index = np.argmax(predictions)
        if max_index == StateMgr.SENTINEL_INDEX:
            self._sentinel_count += 1
            if self._sentinel_count >= StateMgr.SENTINEL_THRES:
                sentinel_active = True
                
                # Sentinel detected, finalize and move to waiting state if processing.
                if self._state == StateMgr.State.PS_PROCESSING:
                    self._state = StateMgr.State.PS_WAITING
                    wmm.finalize_prediction()
                    print('WMM Final:', wmm.get_best_prediction())

                    # Send top phrases to PMM.
                    out_prob = ctypes.c_double()
                    for i in range(NUM_TOP_PHRASES):
                        curr_phrase = wmm.get_next_prediction(ctypes.byref(out_prob))
                        if not curr_phrase:
                            break
                        pmm.add_phrase(curr_phrase, out_prob.value)
                    print('PMM Final:', pmm.get_best_phrase()[0])
        else:
            self._sentinel_count = 0
            
            # Sentinel not detected, reset Word/Phrase Model Managers and move to processing state if waiting.
            if self._state == StateMgr.State.PS_WAITING:
                self._state = StateMgr.State.PS_PROCESSING
                wmm.reset()
                pmm.reset()

        # Wait for debounced sentinel before processing triggers.
        if self._state == StateMgr.State.PS_INIT:
            if sentinel_active:
                self._state = StateMgr.State.PS_PROCESSING
        else:
            # First calculate loss.
            max_value = np.amax(predictions)
            loss = np.sum(predictions) - max_value + (1.0 - max_value)
            # print('LOSS=', loss)

            # Process each letter independently.
            for i in range(0, len(predictions)):
                if self._hysteresis[i] > 0:
                    # In hysteresis phase.
                    self._hysteresis[i] -= 1

                    # Update maximum confidence and update tracked loss.
                    new_conf = predictions[i] * (1.0 - loss)
                    if self._max_conf[i] < new_conf:
                        self._max_conf[i] = new_conf
                        self._loss[i] = loss

                    if self._hysteresis[i] == 0:
                        # Hysteresis completed.
                        # Get maximum X diff and update Word Model Manager.
                        x_diff = self.get_x_diff()
                        if i < 26:
                            ascii_val = i + 65  # Display capital letter
                            wmm.add_letter_prediction(i, self._max_conf[i], self.convert_to_double_prob(x_diff))
                            print('Best:', wmm.get_best_prediction())
                        else:
                            ascii_val = 42  # Display "*" for sentinel

                        # Display diagnostic info.
                        print(chr(ascii_val), " - Conf =", round(self._max_conf[i], 3), "X-diff =", round(x_diff, 3),
                              "Double prob =", round(self.convert_to_double_prob(x_diff), 3),
                              "Delay =", int((time.time() - self._last_time) * 1000), "Loss =", round(self._loss[i], 3))
                        self._last_time = time.time()
                elif predictions[i] > self._thres[i]:
                    if self._state == StateMgr.State.PS_PROCESSING:
                        # Letter 'i' has exceeded probability threshold, it will be triggered after hysteresis.
                        # Increase threshold, initialize hysteresis counter, establish baseline parameters, etc...
                        self._thres[i] = StateMgr.MAX_THRES
                        self._hysteresis[i] = StateMgr.HYSTERESIS_DUR
                        self._max_conf[i] = predictions[i] * (1.0 - loss)
                        self._loss[i] = loss
                        self._suspend[i] = True
                elif self._thres[i] > StateMgr.DEF_THRES:
                    # Don't begin returning to baseline threshold until loss reflects some movement away from current
                    # letter.
                    if self._suspend[i]:
                        if abs(loss - self._loss[i]) >= StateMgr.LOSS_THRES:
                            self._suspend[i] = False
                    else:
                        # Follow linear schedule for returning to baseline threshold.
                        self._thres[i] -= StateMgr.THRES_STEP
                        if self._thres[i] < StateMgr.DEF_THRES:
                            self._thres[i] = StateMgr.DEF_THRES

    def update_x_hist(self):
        """
        Add the current X position to the circular list.
        """
        self._x_hist_index = (self._x_hist_index + 1) % StateMgr.HYSTERESIS_DUR
        self._x_hist[self._x_hist_index] = looseBound[0]

    def get_x_diff(self):
        """
        Calculate the maximum X position difference since start of hysteresis phase.
        :return: maximum X position difference
        """
        oldest_index = (self._x_hist_index + 1) % StateMgr.HYSTERESIS_DUR
        return (max(self._x_hist) - self._x_hist[oldest_index]) / looseBound[2]

    @staticmethod
    def convert_to_double_prob(x_diff):
        """
        Calculate the estimated probability the current letter is a double based on X position diff.
        :param x_diff: maximum X position difference since start of hysteresis phase
        :return: estimated probability of double letter (e.g.: "too" instead of "to")
        """
        if x_diff < StateMgr.DOUBLE_PROB_MIN_THRES:
            return 0.1
        elif x_diff < StateMgr.DOUBLE_PROB_MAX_THRES:
            return 0.1 + (0.8 * ((x_diff - StateMgr.DOUBLE_PROB_MIN_THRES) / StateMgr.DOUBLE_PROB_THRES_RANGE))
        else:
            return 0.9


# ================================================================================
# Functions
# ================================================================================
def profile_entry(label):
    """
    Output message and time since last one for profiling purposes.
    :param label: message to display
    """
    global profileTime

    if PROFILE_ACTIVE:
        print(label, (time.time() - profileTime))
    profileTime = time.time()


def get_main_predictions():
    """
    Run classification against main feature image using main CNN.
    :return: vector of probabilities of each letter (softmax)
    """
    # Pre-process image into feature vector of expected type/range.
    img_float = imgFeatureMain.astype(np.float32)
    data = (img_float - (255.0 / 2.0)) / 255.0
    data = data.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))

    # Run classification using queryPredict tensor.
    query_predict_tensor = mainSess.graph.get_tensor_by_name('queryPredict:0')
    main_predictions = mainSess.run(query_predict_tensor, {'queryData:0': data})
    return np.squeeze(main_predictions)


def get_jz_predictions():
    """
    Run classification against J/Z feature clip using J/Z CNN.
    :return: vector of probabilities of J/Z/neither (softmax)
    """
    global clipArray
    
    # Pre-process image into feature vector of expected type/range.
    img_float = imgFeatureJZ.astype(np.float32)
    data = (img_float - (255.0 / 2.0)) / 255.0
    data = data.reshape((1, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE, 1))

    # Roll clip array and load the new feature.
    # Maintain last CLIP_NUM_CHANNELS feature vectors with newest feature vector last.
    clipArray = np.roll(clipArray, -1, axis=3)
    clipArray[0, :, :, CLIP_NUM_CHANNELS - 1] = data[0, :, :, 0]

    # Run classification using queryPredict tensor.
    query_predict_tensor = jzSess.graph.get_tensor_by_name('queryPredict:0')
    jz_predictions = jzSess.run(query_predict_tensor, {'queryData:0': clipArray})
    return np.squeeze(jz_predictions)


def do_predict(img):
    """
    Implements the core fingerspelling interpretation logic.
    Runs classification of features against the CNNs, merges results, provides diagnostic display, etc.
    :param img: current frame 
    :return: hand acceptance
    """
    global numPredictions
    global lastNumPredictions
    global lastTime
    global rollingPredictions
    global stateMgr
    global wmm
    global pmm

    # Run CNNs on features to get predictions.
    main_predictions = get_main_predictions()
    jz_predictions = get_jz_predictions()

    # Merge predictions.
    predictions = np.empty(NUM_SYMBOLS, dtype=np.float32)
    for i in range(0, len(main_predictions)):
        # Make room for J and Z predictions, shifting to maintain alphabetical order.
        index = i
        if i >= 9:
            index += 1
        if i == 24:
            index = 26
        predictions[index] = main_predictions[i]

    # Scale predictions to accommodate the J/Z prediction values.
    total_clip_predictions = jz_predictions[1] + jz_predictions[2]
    predictions *= (1.0 - total_clip_predictions)

    predictions[9] = jz_predictions[1]
    predictions[25] = jz_predictions[2]

    # Roll prediction history array and load new prediction.
    # Maintain last PREDICT_HIST_SIZE predictions with newest prediction last.
    # Current prediction is smoothed by filtering with a normal probability density function.
    rollingPredictions = np.roll(rollingPredictions, -1, axis=0)
    rollingPredictions[PREDICT_HIST_SIZE - 1, :] = predictions
    curr_predictions = np.sum(rollingPredictions * pdf, axis=0)

    # Handle prediction stats.
    # Currently, we basically just calculate the frame rate.
    numPredictions += 1
    if (time.time() - lastTime) >= 1:
        lastTime = time.time()
        lastNumPredictions = numPredictions
        numPredictions = 0
    cv2.putText(img, str(lastNumPredictions), (580, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 0, 0), 3)

    # Provide display for predictions.
    # For any predictions with >= 0.1 probability, we render the letter, scaling the text based on the probability.
    # (The greater the probability, the larger the letter is rendered.)
    cv2.rectangle(img, (5, 375), (635, 415), (0, 0, 0), cv2.cv.CV_FILLED)
    cv2.rectangle(img, (5, 375), (635, 415), (255, 255, 255))
    for i in range(0, len(curr_predictions)):
        if i < 26:
            ascii_val = i + 65  # Capital letter
        else:
            ascii_val = 42  # *
        if curr_predictions[i] >= 0.1:
            cv2.putText(img, chr(ascii_val), (20 + i * 22, 410), cv2.FONT_HERSHEY_SIMPLEX, curr_predictions[i] * 1.5,
                        (255, 50, 50), 3)

    # Forward latest predictions on to the State Manager.
    stateMgr.process_predictions(curr_predictions)

    # Provide display for latest Word Model Manager prediction.
    cv2.rectangle(img, (5, 415), (635, 445), (0, 0, 0), cv2.cv.CV_FILLED)
    cv2.rectangle(img, (5, 415), (635, 445), (255, 255, 255))
    cv2.putText(img, 'WMM: ' + wmm.get_best_prediction(), (20, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)

    # Provide display for latest Phrase Model Manager prediction.
    cv2.rectangle(img, (5, 445), (635, 475), (0, 0, 0), cv2.cv.CV_FILLED)
    cv2.rectangle(img, (5, 445), (635, 475), (255, 255, 255))
    cv2.putText(img, 'PMM: ' + pmm.get_best_phrase()[0], (20, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)


def do_tuning():
    """
    Implements hysteresis phase in which the main CNN is used to reject false positives and exposure is tuned.
    :return: hand acceptance status
    """
    global tuneModeRemaining
    global tuneNext
    global tuneLastBlowout
    global tuneComplete
    global absoluteExposure
    global devName

    # Run classification on main CNN.
    # The 'A' symbol is used by the cascade classifier for hand location, so here we confirm it against the more robust
    # CNN to quickly rule out false positives.
    main_predictions = get_main_predictions()
    if np.argmax(main_predictions) != 0:
        return HandAcceptance.HA_REJECT

    # If tuning is taking too long, reject hand.
    if tuneModeRemaining == 0:
        return HandAcceptance.HA_REJECT
    tuneModeRemaining -= 1

    # Periodically examine the blowout value last calculated.
    # Tuning is complete once blowout value goes below BLOWOUT_THRES.
    tuneNext -= 1
    if tuneNext == 0:
        tuneNext = BLOWOUT_TEST_FREQ
        if tuneLastBlowout < BLOWOUT_THRES:
            tuneComplete = True
        else:
            # Hand is still blown out.  Decrease exposure using v4l2-ctl.
            # (OpenCV functionality for this is unreliable.)
            absoluteExposure -= 10
            os.system('v4l2-ctl -d ' + devName + ' -c exposure_auto=1 -c exposure_auto_priority=0 ' +
                      '-c exposure_absolute=' + str(absoluteExposure) + ' > /dev/null 2>&1 &')
            print('Decreased exposure to:', absoluteExposure)

    # If tuning was successfully completed, accept hand.
    # Otherwise, continue processing.
    if tuneComplete:
        return HandAcceptance.HA_ACCEPT
    else:
        return HandAcceptance.HA_PROCESSING
    

def get_focus_bound(init_bound):
    """
    Establish a smaller bounding box around the hand within a separate bounding box.
    Focus bounding box is sized to accentuate the center of the hand to provide more useful features for classification.
    :param init_bound: baseline bounding box
    :return: focus bounding box
    """
    return (int(init_bound[2] * BOUND_FOCUS_OFFSET),
            int(init_bound[3] * BOUND_FOCUS_OFFSET),
            int(init_bound[2] * BOUND_FOCUS_FACTOR),
            int(init_bound[3] * BOUND_FOCUS_FACTOR))


def get_tight_bound_in_lb(init_bound):
    """
    Establish a smaller bounding box around the hand within a separate bounding box.
    Tight bounding box is sized to ensure that entire region is hand.
    Similar to get_tight_bound(), expect bounding box is positioned within another "outer" bounding box.
    :param init_bound: baseline bounding box
    :return: tight bounding box
    """
    return (int(init_bound[2] * BOUND_TIGHT_OFFSET),
            int(init_bound[3] * BOUND_TIGHT_OFFSET),
            int(init_bound[2] * BOUND_TIGHT_FACTOR),
            int(init_bound[3] * BOUND_TIGHT_FACTOR))


def convert_distance_to_weight(dist_matrix):
    """
    Translate raw distance-to-edge data into weight using parabolic equation.
    Used to give higher priority to pixels located closer to an edge.
    :param dist_matrix: matrix of distances to edges from the distance transform
    :return: adjusted dist_matrix
    """
    height, width = dist_matrix.shape[:2]
    max_dist = width // 30
    a = -1.0 / (max_dist ** 2)
    dist_matrix = a * (dist_matrix ** 2) + 1.0
    neg_indices = dist_matrix < 0.0
    # noinspection PyUnresolvedReferences
    dist_matrix[neg_indices] = 0.0
    return dist_matrix


def track_hand(img):
    """
    Implement core hand tracking algorithm to keep loose bounding box centered around hand.
    In addition, update feature images for main and J/Z CNNs.
    :param img: current frame
    """
    global looseBound
    global imgFeatureMain
    global imgFeatureJZ
    global tuneLastBlowout
    global sizeAdjustArray
    global sizeAdjustIndex

    # Limit processing to loose bounding box and reduce noise.
    img_roi = img[looseBound[1]:looseBound[1]+looseBound[3], looseBound[0]:looseBound[0]+looseBound[2]]
    img_roi = cv2.GaussianBlur(img_roi, (5, 5), 0)
    profile_entry("PROFILE: TRACKING - FILTERED")

    # Use LUT to generate color-based quality matrix to give priority to pixels with hand colors.
    img_ycc = cv2.cvtColor(img_roi, cv2.COLOR_BGR2YCR_CB)
    lut_matrix = cv2.LUT(img_ycc, lut)
    color_matrix = lut_matrix[:, :, 0] * lut_matrix[:, :, 1] * lut_matrix[:, :, 2]
    color_matrix = cv2.GaussianBlur(color_matrix, (3, 3), 0)
    profile_entry("PROFILE: TRACKING - COLOR MATRIX")

    # Use adaptive canny edge detection and distance transform to generate edge-based quality matrix.
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    success, edges = adaptiveCanny.process(img_gray)
    profile_entry("PROFILE: TRACKING - CANNY")
    if not success:
        print('ERROR: Adaptive canny failed.')
    dist_matrix = cv2.distanceTransform(255 - edges, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
    profile_entry("PROFILE: TRACKING - DIST TRANSFORM")
    dist_matrix = convert_distance_to_weight(dist_matrix)
    profile_entry("PROFILE: TRACKING - EDGE MATRIX")

    # Update blowout if currently tuning.
    # Blowout is calculated as the count of pixels in the tight bounding box that exceed intensity of 200.
    # High values are indicative of over-exposure which can significantly degrade tracking and prediction.
    if currMode == FSIMode.FSI_TUNE:
        tb = get_tight_bound_in_lb(looseBound)
        img_roi = img_gray[tb[1]:tb[1]+tb[3], tb[0]:tb[0]+tb[2]]
        tuneLastBlowout = (img_roi > 200).sum()

    # Calculate master quality matrix, incorporating:
    # 1) Color-based quality matrix (priority to pixels with hand-like colors)
    # 2) Edge-based quality matrix (priority to pixels near edges)
    # 3) Focus weights (priority to pixels near the center)
    # noinspection PyTypeChecker
    master_matrix = color_matrix * dist_matrix * focusWeights
    cv2.normalize(master_matrix, master_matrix, 0.0, 255.0, cv2.NORM_MINMAX)
    profile_entry("PROFILE: TRACKING - MASTER MATRIX")

    # Determine adjusted position of bounding box based on "weighted center of color mass".
    # Find new x position first.
    reduced_cols = cv2.reduce(master_matrix, 0, cv2.cv.CV_REDUCE_SUM)
    master_sum = np.sum(reduced_cols)
    centroid_div = max(master_sum, 1.0)
    val = 0.0
    for i in range(reduced_cols.size):
        val += i * reduced_cols[0, i]
    val /= centroid_div
    new_x = looseBound[0] - ((looseBound[2] // 2) - int(val))

    # Find new y position.
    # Similar to finding x position, but also shift up slightly to give higher priority to top of hand (to avoid feature
    # image including excess wrist/arm).
    reduced_rows = cv2.reduce(master_matrix, 1, cv2.cv.CV_REDUCE_SUM)
    val = 0.0
    for i in range(reduced_rows.size):
        val += i * reduced_rows[i, 0]
    val /= centroid_div
    new_y = looseBound[1] - ((looseBound[3] // 2) - int(val)) - (looseBound[3] // 20)

    # While tuning exposure, perform some additional steps.
    recalc_weights = False
    if not tuneComplete:
        # The following logic attempts to tune the size of the bounding box to make it more consistent.
        # (The initial bounding box returned from the cascade classifier exhibits some undesirable size variance.)
        # First, we step through columns from left to right, stopping after we see 10% of the feature data.
        thres = int(master_sum * 0.1)
        x_left = 0
        for i in range(reduced_cols.size):
            thres -= reduced_cols[0, i]
            if thres <= 0:
                x_left = i
                break
                
        # Same process as above, but proceeding right to left.
        thres = int(master_sum * 0.1)
        x_right = 0
        for i in range(reduced_cols.size - 1, -1, -1):
            thres -= reduced_cols[0, i]
            if thres <= 0:
                x_right = i
                break
        
        # New loose bounding box width/height will be 3 times the range containing 80% of the feature data.
        # Constraining adjustment to within 10% of the original size to avoid catastrophic failure when background is
        # particularly unfavorable.
        new_size = (x_right - x_left) * 3
        if new_size < int(anchorSize * 0.9):
            new_size = int(anchorSize * 0.9)
        elif new_size > int(anchorSize * 1.1):
            new_size = int(anchorSize * 1.1)
        
        # Tracking 10 most recent size adjustment calculations.
        # The average from the last 10 frames of tuning will be used for the final loose bounding box size.
        sizeAdjustArray[sizeAdjustIndex] = new_size
        sizeAdjustIndex = (sizeAdjustIndex + 1) % 10
        
        # Flag for recalculation of the focus weights if size changed.
        if new_size != looseBound[2]:
            recalc_weights = True
    else:
        new_size = looseBound[2]

    # Establish new loose bounding box and recalculate focus weights if needed.
    looseBound = (new_x, new_y, new_size, new_size)
    clip_loose_bound()
    if recalc_weights:
        calc_weights()
    profile_entry("PROFILE: TRACKING - CENTERED")

    # Build feature image for prediction.
    # Using "focus bounding box" which zooms in slightly to focus on the central hand features.
    # Build two different scale images to support the main and J/Z CNNs.
    focus_bound = get_focus_bound(looseBound)
    # noinspection PyUnresolvedReferences
    master_matrix_gray = master_matrix.astype(np.uint8)
    img_quality_base = master_matrix_gray[focus_bound[1]:focus_bound[1]+focus_bound[3],
                                          focus_bound[0]:focus_bound[0]+focus_bound[2]]
    imgFeatureMain = cv2.resize(img_quality_base, (IMAGE_SIZE, IMAGE_SIZE))
    imgFeatureJZ = cv2.resize(img_quality_base, (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE))
    profile_entry("PROFILE: TRACKING - FEATURES")

    # Show loose bounding box on main window and display main feature image in test window.
    cv2.rectangle(img, (looseBound[0], looseBound[1]), (looseBound[0] + looseBound[2], looseBound[1] + looseBound[3]),
                  (255, 0, 0), 2)
    cv2.imshow('Test Display', imgFeatureMain)


def get_tight_bound():
    """
    Establish a smaller bounding box around the hand.
    Tight bounding box is sized to ensure that entire region is hand.
    :return: tight bounding box
    """
    return (int(looseBound[0] + looseBound[2] * BOUND_TIGHT_OFFSET),
            int(looseBound[1] + looseBound[3] * BOUND_TIGHT_OFFSET),
            int(looseBound[2] * BOUND_TIGHT_FACTOR),
            int(looseBound[3] * BOUND_TIGHT_FACTOR))


def build_lut(img, smooth_factor):
    """
    Build the skin tone LUT (Lookup Table) by differencing histograms between the full image and the hand ROI.
    Entries with large positive values correspond to channel values commonly found in the ROI, but not the full image.
    This is later used for hand segmentation/tracking.
    :param img: current frame
    :param smooth_factor: histogram blurring size for reduction of noise/overfitting
    """
    global lut

    # Using YCrCb color space for LUT.
    img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # Use tight bounding box to ensure ROI is entirely hand.
    tb = get_tight_bound()
    img_roi = img_ycc[tb[1]:tb[1]+tb[3], tb[0]:tb[0]+tb[2]]

    # Compute histograms for each channel, on both the full image and ROI.
    full_hists = [cv2.calcHist([img_ycc], [0], None, [256], [0, 256]),
                  cv2.calcHist([img_ycc], [1], None, [256], [0, 256]),
                  cv2.calcHist([img_ycc], [2], None, [256], [0, 256])]
    roi_hists = [cv2.calcHist([img_roi], [0], None, [256], [0, 256]),
                 cv2.calcHist([img_roi], [1], None, [256], [0, 256]),
                 cv2.calcHist([img_roi], [2], None, [256], [0, 256])]

    # Rescale histograms, difference them, and normalize the result to create LUTs for each channel.
    scale_factor = float((tb[2] * tb[3])) / img.size
    diff_hists = []
    for i in range(0, len(full_hists)):
        diff_hists.append(roi_hists[i] - (full_hists[i] * scale_factor))
        diff_hists[i] = cv2.blur(diff_hists[i], (1, smooth_factor))
        cv2.normalize(diff_hists[i], diff_hists[i], 0.0, 1.0, cv2.NORM_MINMAX)

    # Combine normalized difference histograms into the merged LUT.
    lut = cv2.merge((diff_hists[0], diff_hists[1], diff_hists[2]))


def calc_weights():
    """
    Initialize focus weights matrix following parabolic equation.
    Used to give higher priority to pixels located closer to the center of the ROI.
    """
    global focusWeights
    focusWeights = np.empty((looseBound[3], looseBound[2]), np.float32)
    # weight = a * sqr(x - b) + c
    b = looseBound[2] / 2.0
    a = -1.0 / (b * b)
    c = 1.0
    curr_right = looseBound[2] - 1
    curr_bottom = looseBound[3] - 1
    for offset in range(0, (looseBound[2] + 1) // 2):
        val = a * ((offset - b) ** 2) + c
        cv2.rectangle(focusWeights, (offset, offset), (curr_right, curr_bottom), val)
        curr_right -= 1
        curr_bottom -= 1


def clip_loose_bound():
    """
    Update the loose bounding box to ensure that it stays within the boundaries of the image.
    Width and height remain constant; box will be "pushed" as needed to stay inside boundaries.
    """
    global looseBound
    list_bound = list(looseBound)
    if list_bound[0] < 0:
        list_bound[0] = 0
    if list_bound[1] < 0:
        list_bound[1] = 0
    if list_bound[0] + list_bound[2] > 640:
        list_bound[0] = 640 - list_bound[2]
    if list_bound[1] + list_bound[3] > 480:
        list_bound[1] = 480 - list_bound[3]
    looseBound = tuple(list_bound)


def set_loose_bound(rect):
    """
    Establish a larger bounding box around the hand suitable for tracking.
    Loose bounding box is tuned to be large enough that hand remains in the box from frame to frame, even with
    substantial motion.
    :param rect: bounding box of hand as returned from cascade classifier
    """
    global looseBound
    looseBound = (int(rect[0] - (rect[2] * BOUND_LOOSE_OFFSET)),
                  int(rect[1] - (rect[3] * BOUND_LOOSE_OFFSET)),
                  int(rect[2] * BOUND_LOOSE_FACTOR),
                  int(rect[3] * BOUND_LOOSE_FACTOR))


def get_hand_outlier_factor(img_roi):
    """
    Compare YCrCb image statistics between the hand ROI and established statistics from the entire data set to determine
    how far this image deviates from the norm.
    :param img_roi: ROI of located hand
    :return: outlier factor: larger values indicate unusual statistics
    """
    img_ycc = cv2.cvtColor(img_roi, cv2.COLOR_BGR2YCR_CB)
    mean, std_dev = cv2.meanStdDev(img_ycc)
    outlier_factor = 0.0
    for i in range(3):
        print(i, 'Mean', mean[i], 'StdDev', std_dev[i])
        outlier_factor += abs((mean[i] - AVG_MEAN[i]) / AVG_STD_DEV[i])
    print('Hand candidate:', outlier_factor)
    return outlier_factor


def find_hand(img):
    """
    Locate hand in image using a cascade classifier.
    Statistical analysis (the "outlier factor" calculation) is done to quickly eliminate certain false positives.
    If hand is located successfully, we also initialize the focus weights, skin tone LUT, and AdaptiveCanny instance.
    :param img: current frame
    :return: boolean, true if hand is located
    """
    global minNeighbors
    global adaptiveCanny
    global anchorSize
    global looseBound

    # Cascade classifier expects a grayscale, equalized image.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_eq = cv2.equalizeHist(img_gray)
    hands = cascade.detectMultiScale(img_gray_eq, 1.05, minNeighbors, 0, (80, 80))

    # Tighten cascade classifier's neighbor requirement if we're detecting more than one hand.
    found = False
    if len(hands) > 1:
        minNeighbors += 1
    elif len(hands) == 1:
        # Basic hand candidate validation based on YCrCb ROI stats.
        img_roi = img[hands[0][1]:hands[0][1]+hands[0][3], hands[0][0]:hands[0][0]+hands[0][2]]
        if get_hand_outlier_factor(img_roi) < MAX_OUTLIER_FACTOR:
            # Hand candidate accepted.
            found = True

            # Establish loose bounding box around hand and clip to stay inside image.
            set_loose_bound(hands[0])
            clip_loose_bound()
            anchorSize = looseBound[2]

            # Initialize the focus weights matrix.
            calc_weights()

            # Build the skin tone LUT - perform gaussian filter first to ensure similarity with tracking frames.
            img = cv2.GaussianBlur(img, (5, 5), 0)
            build_lut(img, 7)

            # Initialize the AdaptiveCanny instance.
            adaptiveCanny = AdaptiveCanny(0.015, 2.0, True)

            # If tuning was already performed, adjust bounding box size to average of last 10 frames.
            if tuneComplete:
                new_size = sum(sizeAdjustArray) // len(sizeAdjustArray)
                size_diff = (new_size - hands[0][2]) // 2
                looseBound = (looseBound[0] - size_diff, looseBound[1] - size_diff, new_size, new_size)
                clip_loose_bound()
                calc_weights()

    cv2.imshow('Test Display', img_gray)
    return found


def process_args():
    """
    Build a parser, parse the input arguments, then display and return them.
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Launch the Fingerspelling Interpreter.')
    parser.add_argument('device_id', type=int, help='ID of video device')
    return parser.parse_args()


def norm_pdf(x, mu, sigma):
    """
    Normal probability density function implementation.
    (From http://stackoverflow.com/questions/8669235/alternative-for-scipy-stats-norm-pdf.)
    """
    u = (x - mu) / abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * abs(sigma))) * np.exp(-u * u / 2)
    return y


def create_main_graph_and_restore():
    """
    Build graph identical to model used in TrainMainCNN and restore parameters from best training results.
    """
    global mainGraph
    global mainSess

    # Create Tensorflow placeholders/variables.
    print('Building main CNN graph...')
    mainGraph = tf.Graph()
    with mainGraph.as_default():
        train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1),
                                         name='trainDataNode')
        train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS),
                                           name='trainLabelsNode')
        validation_data_node = tf.zeros((VALIDATION_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
        test_data_node = tf.zeros((TEST_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))

        conv1_weights = tf.Variable(
            tf.truncated_normal([KERNEL_SIZE, KERNEL_SIZE, 1, CONV_DEPTH],
                                stddev=math.sqrt(1.0 / (5 * 5)),
                                seed=None), name='conv1_weights')
        conv1_biases = tf.Variable(tf.zeros([CONV_DEPTH]), name='conv1_biases')
        conv2_weights = tf.Variable(
                tf.truncated_normal([KERNEL_SIZE, KERNEL_SIZE, CONV_DEPTH, (CONV_DEPTH * 2)],
                                    stddev=math.sqrt(2.0 / (5 * 5 * CONV_DEPTH)),
                                    seed=None), name='conv2_weights')
        conv2_biases = tf.Variable(tf.zeros([(CONV_DEPTH * 2)]), name='conv2_biases')
        conv3_weights = tf.Variable(
                tf.truncated_normal([KERNEL_SIZE, KERNEL_SIZE, (CONV_DEPTH * 2), (CONV_DEPTH * 4)],
                                    stddev=math.sqrt(2.0 / (5 * 5 * CONV_DEPTH * 2)),
                                    seed=None), name='conv3_weights')
        conv3_biases = tf.Variable(tf.zeros([(CONV_DEPTH * 4)]), name='conv3_biases')
        conv4_weights = tf.Variable(
                tf.truncated_normal([KERNEL_SIZE, KERNEL_SIZE, (CONV_DEPTH * 4), (CONV_DEPTH * 8)],
                                    stddev=math.sqrt(2.0 / (5 * 5 * CONV_DEPTH * 4)),
                                    seed=None), name='conv4_weights')
        conv4_biases = tf.Variable(tf.zeros([(CONV_DEPTH * 8)]), name='conv4_biases')
        fcl_fanin = (IMAGE_SIZE // 16) * (IMAGE_SIZE // 16) * CONV_DEPTH * 8
        fc1_weights = tf.Variable(
                tf.truncated_normal([fcl_fanin, FC_DEPTH],
                                    stddev=math.sqrt(2.0 / fcl_fanin),
                                    seed=None), name='fc1_weights')
        fc1_biases = tf.Variable(tf.zeros([FC_DEPTH]), name='fc1_biases')
        fc2_weights = tf.Variable(
                tf.truncated_normal([FC_DEPTH, NUM_LABELS],
                                    stddev=math.sqrt(2.0 / FC_DEPTH),
                                    seed=None), name='fc2_weights')
        fc2_biases = tf.Variable(tf.zeros([NUM_LABELS]), name='fc2_biases')

        # We will replicate the model structure for the training subgraph, as well as the evaluation subgraphs, while
        # sharing the trainable parameters.
        def model(data, train=False):
            """
            Build the model for the main CNN.
            :param data: input data node (e.g.: training minibatch, validation, or test data)
            :param train: boolean, true indicates training subgraph, thus apply dropout
            :return: model output logits
            """
            conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            pool_shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

            if train:
                hidden = tf.nn.dropout(hidden, KEEP_PROB, seed=SEED)

            return tf.matmul(hidden, fc2_weights) + fc2_biases

        # Training computation: logits + cross-entropy loss.
        logits = model(train_data_node, True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        # Add the regularization term to the loss.
        loss += REG_FACTOR * regularizers

        # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule.
        learning_rate = tf.train.exponential_decay(
                LR_INIT,  # Base learning rate.
                batch * BATCH_SIZE,  # Current index into the dataset.
                NUM_IMAGES,  # Decay step.
                LR_DECAY,  # Decay rate.
                staircase=True)

        # Use simple momentum for the optimization.
        tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss,
                                                                global_step=batch,
                                                                name='optimizer')

        # Predictions for the minibatch, validation set and test set.
        tf.nn.softmax(logits)

        # We'll compute them only once in a while by calling their {eval()} method.
        tf.nn.softmax(model(validation_data_node))
        tf.nn.softmax(model(test_data_node))

        # New query nodes
        query_data_node = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, 1), name='queryData')
        response_logits = model(query_data_node, False)
        tf.nn.softmax(response_logits, name='queryPredict')

        # Perform session initialization and restore checkpoint.
        print('Restoring model variables from checkpoint...')
        saver = tf.train.Saver()
        mainSess = tf.Session()
        mainSess.run(tf.initialize_all_variables())
        saver.restore(mainSess, 'model/main-cnn/train-vars-best-main')


def create_jz_graph_and_restore():
    """
    Build graph identical to model used in TrainMainCNN and restore parameters from best training results.
    """
    global jzGraph
    global jzSess

    # Create Tensorflow placeholders/variables.
    print('Building J/Z CNN graph...')
    jzGraph = tf.Graph()
    with jzGraph.as_default():
        train_data_node = tf.placeholder(tf.float32, shape=(CLIP_BATCH_SIZE, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE,
                                                            CLIP_NUM_CHANNELS))
        train_labels_node = tf.placeholder(tf.float32, shape=(CLIP_BATCH_SIZE, CLIP_NUM_LABELS))
        validation_data_node = tf.zeros((CLIP_VALIDATION_SIZE, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE, CLIP_NUM_CHANNELS))
        test_data_node = tf.zeros((CLIP_TEST_SIZE, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE, CLIP_NUM_CHANNELS))
        
        conv1_stddev = math.sqrt(1.0 / (CLIP_KERNEL_SIZE * CLIP_KERNEL_SIZE * CLIP_NUM_CHANNELS))
        conv1_weights = tf.Variable(
                tf.truncated_normal([CLIP_KERNEL_SIZE, CLIP_KERNEL_SIZE, CLIP_NUM_CHANNELS, CLIP_CONV_DEPTH],
                                    stddev=conv1_stddev,
                                    seed=SEED), name='conv1_weights')
        conv1_biases = tf.Variable(tf.constant(conv1_stddev, shape=[CLIP_CONV_DEPTH]), name='conv1_biases')
        conv2_stddev = math.sqrt(2.0 / (CLIP_KERNEL_SIZE * CLIP_KERNEL_SIZE * CLIP_CONV_DEPTH))
        conv2_weights = tf.Variable(
                tf.truncated_normal([CLIP_KERNEL_SIZE, CLIP_KERNEL_SIZE, CLIP_CONV_DEPTH, (CLIP_CONV_DEPTH * 2)],
                                    stddev=conv2_stddev,
                                    seed=SEED), name='conv2_weights')
        conv2_biases = tf.Variable(tf.constant(conv2_stddev, shape=[(CLIP_CONV_DEPTH * 2)]), name='conv2_biases')
        conv3_stddev = math.sqrt(2.0 / (CLIP_KERNEL_SIZE * CLIP_KERNEL_SIZE * CLIP_CONV_DEPTH * 2))
        conv3_weights = tf.Variable(
                tf.truncated_normal([CLIP_KERNEL_SIZE, CLIP_KERNEL_SIZE, (CLIP_CONV_DEPTH * 2), (CLIP_CONV_DEPTH * 4)],
                                    stddev=conv3_stddev,
                                    seed=SEED), name='conv3_weights')
        conv3_biases = tf.Variable(tf.constant(conv3_stddev, shape=[(CLIP_CONV_DEPTH * 4)]), name='conv3_biases')
        conv4_stddev = math.sqrt(2.0 / (CLIP_KERNEL_SIZE * CLIP_KERNEL_SIZE * CLIP_CONV_DEPTH * 4))
        conv4_weights = tf.Variable(
                tf.truncated_normal([CLIP_KERNEL_SIZE, CLIP_KERNEL_SIZE, (CLIP_CONV_DEPTH * 4), (CLIP_CONV_DEPTH * 8)],
                                    stddev=conv4_stddev,
                                    seed=SEED), name='conv4_weights')
        conv4_biases = tf.Variable(tf.constant(conv4_stddev, shape=[(CLIP_CONV_DEPTH * 8)]), name='conv4_biases')
        fcl_fanin = (CLIP_IMAGE_SIZE // 16) * (CLIP_IMAGE_SIZE // 16) * CLIP_CONV_DEPTH * 8
        fc1_stddev = math.sqrt(2.0 / fcl_fanin)
        fc1_weights = tf.Variable(
                tf.truncated_normal(
                        [fcl_fanin, CLIP_FC_DEPTH],
                        stddev=fc1_stddev,
                        seed=SEED), name='fc1_weights')
        fc1_biases = tf.Variable(tf.constant(fc1_stddev, shape=[CLIP_FC_DEPTH]), name='fc1_biases')
        fc2_stddev = math.sqrt(2.0 / CLIP_FC_DEPTH)
        fc2_weights = tf.Variable(
                tf.truncated_normal([CLIP_FC_DEPTH, CLIP_NUM_LABELS],
                                    stddev=fc2_stddev,
                                    seed=SEED), name='fc2_weights')
        fc2_biases = tf.Variable(tf.constant(fc2_stddev, shape=[CLIP_NUM_LABELS]), name='fc2_biases')

        # We will replicate the model structure for the training subgraph, as well as the evaluation subgraphs, while
        # sharing the trainable parameters.
        def model(data, train=False):
            """
            Build the model for the main CNN.
            :param data: input data node (e.g.: training minibatch, validation, or test data)
            :param train: boolean, true indicates training subgraph, thus apply dropout
            :return: model output logits
            """
            conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            pool_shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

            if train:
                hidden = tf.nn.dropout(hidden, CLIP_KEEP_PROB, seed=SEED)

            return tf.matmul(hidden, fc2_weights) + fc2_biases

        # Training computation: logits + cross-entropy loss.
        logits = model(train_data_node, True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        # Add the regularization term to the loss.
        loss += CLIP_REG_FACTOR * regularizers

        # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule.
        learning_rate = tf.train.exponential_decay(
                CLIP_LR_INIT,  # Base learning rate.
                batch * CLIP_BATCH_SIZE,  # Current index into the dataset.
                CLIP_NUM_IMAGES,  # Decay step.
                CLIP_LR_DECAY,  # Decay rate.
                staircase=True)

        # Use simple momentum for the optimization.
        tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss,
                                                                global_step=batch)

        # Predictions for the minibatch, validation set and test set.
        tf.nn.softmax(logits)

        # We'll compute them only once in a while by calling their {eval()} method.
        tf.nn.softmax(model(validation_data_node))
        tf.nn.softmax(model(test_data_node))

        # New query nodes
        query_data_node = tf.placeholder(tf.float32, shape=(1, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE, CLIP_NUM_CHANNELS),
                                         name='queryData')
        response_logits = model(query_data_node, False)
        tf.nn.softmax(response_logits, name='queryPredict')

        # Perform session initialization and restore checkpoint.
        print('Restoring model variables from checkpoint...')
        saver = tf.train.Saver()
        jzSess = tf.Session()
        jzSess.run(tf.initialize_all_variables())
        saver.restore(jzSess, 'model/jz-cnn/train-vars-best-jz')


def process_image(img):
    """
    Perform all processing of current frame.
    Implements a simple state machine driven by the FSI mode.
    FSI_INIT: Initial state.
    FSI_DELAY: Delay for delayDur seconds.
    FSI_FIND: Locate hand in image.
    FSI_TUNE: Hysteresis phase in which the main CNN is used to reject false positives and exposure is tuned.
    FSI_TRACK: Main phase in which hand is tracked and fingerspelling is interpreted.
    :param img: current frame
    """
    global currMode
    global delayDur
    global startTime
    global tuneModeRemaining
    global tuneNext
    global tuneLastBlowout

    if currMode == FSIMode.FSI_INIT:
        startTime = time.time()
        print('Waiting for webcam auto-adjustment to complete...')
        currMode = FSIMode.FSI_DELAY
        delayDur = 5
    elif currMode == FSIMode.FSI_DELAY:
        if time.time() - startTime >= delayDur:
            print('Finding hand...')
            currMode = FSIMode.FSI_FIND
    elif currMode == FSIMode.FSI_FIND:
        if find_hand(img):
            if tuneComplete:
                print('Tracking hand...')
                currMode = FSIMode.FSI_TRACK
            else:
                print('Tuning exposure...')
                currMode = FSIMode.FSI_TUNE
                tuneModeRemaining = MAX_TUNING_FRAMES
                tuneNext = BLOWOUT_TEST_FREQ
                tuneLastBlowout = 0
    elif currMode == FSIMode.FSI_TUNE:
        track_hand(img)
        hand_accept = do_tuning()
        if hand_accept != HandAcceptance.HA_PROCESSING:
            if hand_accept == HandAcceptance.HA_ACCEPT:
                print('Tuning successful.')
            elif hand_accept == HandAcceptance.HA_REJECT:
                print('Tuning failed.')
            currMode = FSIMode.FSI_DELAY
            delayDur = 1
            startTime = time.time()
    elif currMode == FSIMode.FSI_TRACK:
        profile_entry("PROFILE: START TRACK")
        track_hand(img)
        profile_entry("PROFILE: START PREDICT")
        do_predict(img)
        profile_entry("PROFILE: END TRACK")


def main():
    global pdf
    global wmm
    global pmm
    global devName
    global stateMgr

    # Attempt to retrieve/validate parameters.
    args = process_args()

    # Build probability density function to smooth rolling J/Z predictions.
    for i in range(0, PREDICT_HIST_SIZE):
        pdf[i] = norm_pdf(i, 3.5, 1.5)

    print('Initializing Word Model Manager...')
    wmm = WordModelMgr()
    if not wmm.initialize():
        print('ERROR: WMM initialization failed.')
        return

    print('Initializing Phrase Model Manager...')
    pmm = PhraseModelMgr()

    # At the time of this development effort, Tensorflow does not easily support loading a model from file and restoring
    # saved parameters.  Instead, rebuilding both CNN graphs manually and restoring from saved checkpoints.
    create_main_graph_and_restore()
    create_jz_graph_and_restore()
    profile_entry("PROFILE: GRAPHS RESTORED")

    # Establish video capture on requested device.
    vcap = cv2.VideoCapture(args.device_id)
    if not vcap.isOpened():
        print('ERROR: Cannot open specified video device.')
        return
    devName = '/dev/video' + str(args.device_id)
    profile_entry("PROFILE: VIDEO CAPTURE ESTABLISHED")

    # Establish and position windows.
    cv2.namedWindow('Test Display', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Test Display', 40, 500)
    cv2.namedWindow('Main Display')
    cv2.moveWindow('Main Display', 640, 40)

    # Initialize the State Manager.
    stateMgr = StateMgr()

    # Executive loop - process images until ESC key is pressed.
    update_exposure = True
    absolute_exposure = absoluteExposure
    while True:
        # Read frame, process it, display current mode, and display.
        # profile_entry("PROFILE: LOOP START")
        ret_val, img = vcap.read()
        # profile_entry("PROFILE: FRAME GRABBED")
        process_image(img)
        # profile_entry("PROFILE: PROCESSED")
        cv2.rectangle(img, (5, 5), (310, 30), (0, 0, 0), cv2.cv.CV_FILLED)
        cv2.putText(img, fsiModeString[currMode], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('Main Display', img)
        # profile_entry("PROFILE: DISPLAYED")

        # Handle user input.
        key = (cv2.waitKey(1) % 256)
        if key == 27:
            break
        elif key == ord('+'):
            update_exposure = True
            absolute_exposure += 25
            if absolute_exposure > 500:
                absolute_exposure = 500
        elif key == ord('-'):
            update_exposure = True
            absolute_exposure -= 25
            if absolute_exposure < 100:
                absolute_exposure = 100
        elif key == ord('1'):
            wmm.dump_candidates()
        elif key == ord('2'):
            pmm.dump_phrases()

        # Process manual exposure updates.
        if update_exposure:
            os.system('v4l2-ctl -d ' + devName + ' -c exposure_auto=1 -c exposure_auto_priority=0 ' +
                      '-c exposure_absolute=' + str(absolute_exposure) + ' > /dev/null 2>&1 &')
            update_exposure = False

    # Cleanup.
    vcap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
