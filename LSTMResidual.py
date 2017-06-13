import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from random import shuffle
from random import sample


class LSTMResidual(object):

    # Start with the basic data that is needed to set the file
    # up. Will try with 2 input files and see what happens
    # choose the 10th instrument which is ZB and use that for predicition

    def __init__(self, ninstr, seqdepth, batchsize=5000):
        """
        Set up all the tensorflow variables
        :param ninstr: int, number of instruments in the data set
        :param seqdepth: int, number of timesteps looking back
        :param batchsize: int, how many to train with
        """

        self.ninstr = ninstr
        self.seqdepth = seqdepth
        self.batchsize = batchsize

        # Input: batch size x seq_length x input_dimension
        # Output: one hot encoding of ZB residual into 5 classes
        self.classvalues = 5
        self.data    = tf.placeholder(tf.float32, [None, self.seqdepth, self.ninstr], name="Data")
        self.target  = tf.placeholder(tf.float32, [None, self.classvalues], name="Target")

        # Figure out self.val and self.last shapes
        num_hidden = seqdepth*ninstr
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)
        self.val, self.state = tf.nn.dynamic_rnn(self.cell, self.data, dtype=tf.float32 )
        self.val = tf.transpose(self.val, [1, 0, 2])
        self.last = tf.gather(self.val, int(self.val.get_shape()[0])-1)

        # Final softmaxish layer
        self.weight = tf.Variable(tf.truncated_normal([num_hidden, int(self.target.get_shape()[1])]))
        self.bias   = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))

        # Sum over errors in all predictions
        self.prediction = tf.matmul(self.last, self.weight) + self.bias
        self.xent      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.target))

        # Add cross entropy etc
        self.optimizer = tf.train.AdamOptimizer()
        self.mini = self.optimizer.minimize(self.xent)

        self.mistakes = tf.not_equal(tf.argmax(self.prediction, 1), tf.argmax(self.target, 1))
        self.errorrate = tf.reduce_mean(tf.cast(self.mistakes, dtype=tf.float32))

        # Also categorize mistakes into good (same sign outcome) and bad (different sign outcome)


