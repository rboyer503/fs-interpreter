# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Parameterized script for training the J/Z CNN for interpreting the American Manual Alphabet letters J and Z which
involve motion.
Adapted from the Google Tensorflow MNIST example.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import gzip
import sys
import time
import math
import argparse


# Constants
DATA_DIRECTORY = 'data/'
MODEL_DIRECTORY = 'model/jz-cnn/'

IMAGE_SIZE = 64
NUM_CHANNELS = 8
PIXEL_DEPTH = 255
NUM_CLIPS = 7200
NUM_LABELS = 3
VALIDATION_SIZE = 100
TEST_SIZE = 1440
SEED = None  # 66478
KERNEL_SIZE_C1 = 3
KERNEL_SIZE_C2 = 3
KERNEL_SIZE_C3 = 3
KERNEL_SIZE_C4 = 3
NUM_EPOCHS = 40


def process_args():
    """
    Build a parser, parse the input arguments, then display and return them.
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train J/Z CNN (update model variables).')
    parser.add_argument('base_conv_depth', type=int, help='base depth for conv layers')
    parser.add_argument('fcl_depth', type=int, help='depth of fully-connected layer')
    parser.add_argument('lr_init', type=float, help='initial learning rate')
    parser.add_argument('lr_decay', type=float, help='learning rate decay')
    parser.add_argument('keep_prob', type=float, help='dropout keep probability')
    parser.add_argument('reg_factor', type=float, help='L2 regularization factor')
    parser.add_argument('b_size', type=int, help='batch size')

    args = parser.parse_args()
    print('Base conv depth:', args.base_conv_depth)
    print('FCL depth:', args.fcl_depth)
    print('LR init:', args.lr_init)
    print('LR decay:', args.lr_decay)
    print('Keep prob:', args.keep_prob)
    print('Reg factor:', args.reg_factor)
    print('Batch size:', args.b_size)
    return args


def extract_data(filename, num_clips):
    """
    Extract motion clips from a raw gzip data file into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    :param filename: path to raw gzip data file
    :param num_clips: number of motion clips in file
    :return: 4D tensor
    """
    print('Extracting data:', filename)
    with gzip.open(filename) as bytestream:
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS * num_clips)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        # noinspection PyUnresolvedReferences
        data = data.reshape(num_clips * NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, 1)
        ret_data = np.empty((num_clips, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
        for clipIndex in range(0, num_clips):
            for channelIndex in range(0, NUM_CHANNELS):
                ret_data[clipIndex, :, :, channelIndex] = data[(clipIndex * NUM_CHANNELS) + channelIndex, :, :, 0]
        return ret_data


def extract_labels(filename, num_clips):
    """
    Extract labels from a raw gzip labels file into a 1-hot matrix [clip index, label index].
    :param filename: path to raw gzip labels file
    :param num_clips: number of motion clips labelled in file
    :return: 2D tensor (1-hot matrix)
    """
    print('Extracting labels:', filename)
    with gzip.open(filename) as bytestream:
        buf = bytestream.read(num_clips)
        labels = np.frombuffer(buf, dtype=np.uint8)

    # Convert to dense 1-hot representation.
    return (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    :param predictions: model predictions
    :param labels: actual labels
    :return: error rate (ratio of incorrectly classified samples)
    """
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])


def main():
    # Attempt to retrieve/validate parameters.
    args = process_args()

    # Build paths to data and label files (for both training and test sets).
    data_filename = DATA_DIRECTORY + "clipData-train.raw.gz"
    labels_filename = DATA_DIRECTORY + "clipLabels-train.raw.gz"
    data_test_filename = DATA_DIRECTORY + "clipData-test.raw.gz"
    labels_test_filename = DATA_DIRECTORY + "clipLabels-test.raw.gz"

    # Extract training and test data and labels.
    train_data = extract_data(data_filename, NUM_CLIPS)
    train_labels = extract_labels(labels_filename, NUM_CLIPS)
    test_data = extract_data(data_test_filename, TEST_SIZE)
    test_labels = extract_labels(labels_test_filename, TEST_SIZE)

    # Shuffle the data.
    perm = np.arange(NUM_CLIPS)
    np.random.shuffle(perm)
    train_data = train_data[perm]
    train_labels = train_labels[perm]
    perm = np.arange(TEST_SIZE)
    np.random.shuffle(perm)
    curr_train_data = train_data[:, :, :, :]
    curr_train_labels = train_labels[:]

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :, :, :]
    validation_labels = train_labels[:VALIDATION_SIZE]
    num_epochs = NUM_EPOCHS
    train_size = curr_train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each training step using the {feed_dict} argument
    # to the Run() call below.
    train_data_node = tf.placeholder(
            tf.float32,
            shape=(args.b_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(args.b_size, NUM_LABELS))

    # For the validation and test data, we'll just hold the entire dataset in one constant node.
    validation_data_node = tf.constant(validation_data)
    test_data_node = tf.constant(test_data)

    # The variables below hold all the trainable weights. They are passed an initial value which will be assigned when
    # when we call: {tf.initialize_all_variables().run()}.
    conv1_stddev = math.sqrt(1.0 / (KERNEL_SIZE_C1 * KERNEL_SIZE_C1 * NUM_CHANNELS))
    conv1_weights = tf.Variable(
            tf.truncated_normal([KERNEL_SIZE_C1, KERNEL_SIZE_C1, NUM_CHANNELS, args.base_conv_depth],
                                stddev=conv1_stddev,
                                seed=SEED), name='conv1_weights')
    conv1_biases = tf.Variable(tf.constant(conv1_stddev, shape=[args.base_conv_depth]), name='conv1_biases')
    conv2_stddev = math.sqrt(2.0 / (KERNEL_SIZE_C2 * KERNEL_SIZE_C2 * args.base_conv_depth))
    conv2_weights = tf.Variable(
            tf.truncated_normal([KERNEL_SIZE_C2, KERNEL_SIZE_C2, args.base_conv_depth, (args.base_conv_depth * 2)],
                                stddev=conv2_stddev,
                                seed=SEED), name='conv2_weights')
    conv2_biases = tf.Variable(tf.constant(conv2_stddev, shape=[(args.base_conv_depth * 2)]), name='conv2_biases')
    conv3_stddev = math.sqrt(2.0 / (KERNEL_SIZE_C3 * KERNEL_SIZE_C3 * args.base_conv_depth * 2))
    conv3_weights = tf.Variable(
            tf.truncated_normal([KERNEL_SIZE_C3, KERNEL_SIZE_C3, (args.base_conv_depth * 2),
                                 (args.base_conv_depth * 4)],
                                stddev=conv3_stddev,
                                seed=SEED), name='conv3_weights')
    conv3_biases = tf.Variable(tf.constant(conv3_stddev, shape=[(args.base_conv_depth * 4)]), name='conv3_biases')
    conv4_stddev = math.sqrt(2.0 / (KERNEL_SIZE_C4 * KERNEL_SIZE_C4 * args.base_conv_depth * 4))
    conv4_weights = tf.Variable(
            tf.truncated_normal([KERNEL_SIZE_C4, KERNEL_SIZE_C4, (args.base_conv_depth * 4),
                                 (args.base_conv_depth * 8)],
                                stddev=conv4_stddev,
                                seed=SEED), name='conv4_weights')
    conv4_biases = tf.Variable(tf.constant(conv4_stddev, shape=[(args.base_conv_depth * 8)]), name='conv4_biases')
    fcl_fanin = (IMAGE_SIZE // 16) * (IMAGE_SIZE // 16) * args.base_conv_depth * 8
    fc1_stddev = math.sqrt(2.0 / fcl_fanin)
    fc1_weights = tf.Variable(
            tf.truncated_normal(
                    [fcl_fanin, args.fcl_depth],
                    stddev=fc1_stddev,
                    seed=SEED), name='fc1_weights')
    fc1_biases = tf.Variable(tf.constant(fc1_stddev, shape=[args.fcl_depth]), name='fc1_biases')
    fc2_stddev = math.sqrt(2.0 / args.fcl_depth)
    fc2_weights = tf.Variable(
            tf.truncated_normal([args.fcl_depth, NUM_LABELS],
                                stddev=fc2_stddev,
                                seed=SEED), name='fc2_weights')
    fc2_biases = tf.Variable(tf.constant(fc2_stddev, shape=[NUM_LABELS]), name='fc2_biases')

    # We will replicate the model structure for the training subgraph, as well as the evaluation subgraphs, while
    # sharing the trainable parameters.
    def model(data, train=False):
        """
        Build the model for the main CNN.
        :param data: input data node (e.g.: training minibatch, validation, or test data)
        :param train: boolean, true indicates training subgraph, thus apply dropout
        :return: model output logits
        """

        # =========================
        # First convolutional layer
        # =========================
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has the same size as the input). Note that
        # {strides} is a 4D array whose shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and ReLU non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of the data. Here we have a pooling window
        # of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # ==========================
        # Second convolutional layer
        # ==========================
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # =========================
        # Third convolutional layer
        # =========================
        conv = tf.nn.conv2d(pool,
                            conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # ==========================
        # Fourth convolutional layer
        # ==========================
        conv = tf.nn.conv2d(pool,
                            conv4_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # ===========================
        # First fully-connected layer
        # ===========================
        # Reshape the feature map cuboid into a 2D matrix to feed it to the fully-connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        # Add a 50% dropout during training only.
        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, args.keep_prob, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += args.reg_factor * regularizers

    # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule.
    learning_rate = tf.train.exponential_decay(
            args.lr_init,  # Base learning rate.
            batch * args.b_size,  # Current index into the dataset.
            train_size,  # Decay step.
            args.lr_decay,  # Decay rate.
            staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    validation_prediction = tf.nn.softmax(model(validation_data_node))
    test_prediction = tf.nn.softmax(model(test_data_node))

    # New query nodes
    query_data_node = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='queryData')
    response_logits = model(query_data_node, False)
    query_prediction = tf.nn.softmax(response_logits, name='queryPredict')

    # Create a local session to run all training, validation, and testing.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()

        # Restore checkpoint if available.
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_DIRECTORY)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(s, ckpt.model_checkpoint_path)

        curr_epoch = 0
        best_test_error = 100.0
        print('Initialized!')

        # Loop through training steps.
        for step in xrange(num_epochs * train_size // args.b_size):
            if ((step * args.b_size) // (train_size - args.b_size)) > curr_epoch:
                # Complete an epoch - reshuffle data.
                curr_epoch += 1
                perm = np.arange(NUM_CLIPS)
                np.random.shuffle(perm)
                train_data = train_data[perm]
                train_labels = train_labels[perm]
                curr_train_data = train_data[:, :, :, :]
                curr_train_labels = train_labels[:]

            # Compute the offset of the current minibatch in the data.
            offset = (step * args.b_size) % (train_size - args.b_size)
            batch_data = curr_train_data[offset:(offset + args.b_size), :, :, :]
            batch_labels = curr_train_labels[offset:(offset + args.b_size)]

            # This dictionary maps the batch data (as a numpy array) to the node in the graph is should be fed to.
            # Run the graph and fetch some of the nodes.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            _, l, lr, predictions = s.run(
                    [optimizer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)

            # Every 50th step, run validation, output status update, etc...
            if step % 50 == 0:
                print('Epoch %.2f' % (float(step) * args.b_size / train_size))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' %
                      error_rate(validation_prediction.eval(), validation_labels))

                # Track best error rate against test data set.
                # Save current checkpoint and update best checkpoint.
                test_error = error_rate(test_prediction.eval(), test_labels)
                if test_error <= best_test_error:
                    best_test_error = test_error
                    saver.save(s, MODEL_DIRECTORY + 'train-vars-best', write_meta_graph=False)
                print('Test error: %.1f%%' % test_error)
                print(time.ctime())
                sys.stdout.flush()
                saver.save(s, MODEL_DIRECTORY + 'train-vars', write_meta_graph=False)

        # Final test data set results.
        test_error = error_rate(test_prediction.eval(), test_labels)
        if test_error <= best_test_error:
            best_test_error = test_error
            saver.save(s, MODEL_DIRECTORY + 'train-vars-best', write_meta_graph=False)
        print('Final test error: %.1f%%' % test_error)
        print('Best test error: %.1f%%' % best_test_error)
        # tf.train.write_graph(s.graph_def, '/tmp/models', 'train1.pb', False)
        saver.save(s, MODEL_DIRECTORY + 'train-vars', write_meta_graph=False)


if __name__ == '__main__':
    main()
