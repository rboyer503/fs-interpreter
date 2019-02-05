"""
Phrase Model Manager
Defines class to manage RNN LTSM model trained on PTB corpus for evaluation of perplexity of proposed phrases.
"""

import tensorflow as tf
import numpy as np
from operator import itemgetter

# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import seq2seq
import reader


# Constants
DATA_DIRECTORY = 'data/'
MODEL_DIRECTORY = 'model/pmm-rnn/'


class PTBModel(object):
    """
    Class implementing the PTB model.
    """

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        states = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
                states.append(state)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        logits = tf.nn.xw_plus_b(output,
                                 tf.get_variable("softmax_w", [size, vocab_size]),
                                 tf.get_variable("softmax_b", [vocab_size]))
        loss = seq2seq.sequence_loss_by_example([logits],
                                                [tf.reshape(self._targets, [-1])],
                                                [tf.ones([batch_size * num_steps])],
                                                vocab_size)
        self._cost = tf.reduce_sum(loss) / batch_size
        self._final_state = states[-1]

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state


class EvalConfigSmall(object):
    """
    Small evaluation configuration.
    """
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 1
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    vocab_size = 10002


class EvalConfigMedium(object):
    """
    Medium evaluation configuration.
    """
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 1
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 1
    vocab_size = 10002


class PhraseModelMgr(object):
    def __init__(self):
        self.phrases = []
        self.best_phrase = None
        self.word_to_id = reader.build_vocab(DATA_DIRECTORY + 'ptb.train.txt')
        self.pmm_graph = tf.Graph()
        with self.pmm_graph.as_default():
            with tf.variable_scope("model"):
                self.pmm_model = PTBModel(is_training=False, config=EvalConfigMedium())
            saver = tf.train.Saver()
            self.pmm_session = tf.Session()
            self.pmm_session.run(tf.initialize_all_variables())
            saver.restore(self.pmm_session, MODEL_DIRECTORY + 'train-vars-rnn')

    def add_phrase(self, phrase, confidence):
        # Parse phrase
        input_ids = []
        penalty = 1.0
        for word in phrase.lower().split():
            if word in self.word_to_id:
                input_ids.append(self.word_to_id[word])
            else:
                input_ids.append(self.word_to_id["<unk>"])
                penalty *= 1.5
        input_ids.append(self.word_to_id["<eos>"])
        # print input_ids

        # Calculate perplexity
        costs = 0.0
        iters = 0
        state = self.pmm_model.initial_state.eval(session=self.pmm_session)
        for step, (x, y) in enumerate(reader.ptb_iterator(input_ids, self.pmm_model.batch_size,
                                                          self.pmm_model.num_steps)):
            cost, state, = self.pmm_session.run([self.pmm_model.cost, self.pmm_model.final_state],
                                                {self.pmm_model.input_data: x,
                                                 self.pmm_model.targets: y,
                                                 self.pmm_model.initial_state: state})
            costs += cost
            iters += self.pmm_model.num_steps
        perplexity = np.exp(costs / iters) ** penalty

        self.phrases.append((phrase, confidence, perplexity))
        self.best_phrase = None
        return perplexity

    def get_best_phrase(self):
        if self.best_phrase is not None:
            return self.best_phrase

        if not self.phrases:
            return "", 0.0

        max_perp = max(self.phrases, key=itemgetter(2))[2] * 2

        max_adj_conf = 0.0
        for (phrase, conf, perp) in self.phrases:
            adj_conf = conf * (1.0 - (perp / max_perp))
            # print('Phrase candidate: %s, Conf=%0.4f' % (phrase, adj_conf))
            if adj_conf > max_adj_conf:
                self.best_phrase = (phrase, adj_conf)
                max_adj_conf = adj_conf

        return self.best_phrase

    def reset(self):
        self.phrases = []
        self.best_phrase = None

    def dump_phrases(self):
        for (phrase, conf, perp) in self.phrases:
            print('%s: Conf=%.2f, Perp=%.2f' % (phrase, conf, perp))
