import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from random import shuffle
from random import sample
from TestDB import FileDB as FileDB
from matplotlib import pyplot as plt


class LSTMResidual(object):

    # Start with the basic data that is needed to set the file
    # up. Will try with 2 input files and see what happens
    # choose the 10th instrument which is ZB and use that for predicition

    def __init__(self, ninstr, seqdepth, batchsize=5000, classvalues=6):
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
        # Output: one hot encoding of  residual into 6 classes
        self.classvalues = classvalues
        self.data    = tf.placeholder(tf.float32, [None, self.seqdepth, self.ninstr], name="Data")
        self.target  = tf.placeholder(tf.float32, [None, self.ninstr], name="Target")

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
        self.prediction1 = tf.matmul(self.last, self.weight) + self.bias
        self.differencevec = tf.subtract(self.prediction1, self.target, name="diffvec")
        self.diffsq = tf.norm(self.differencevec, name="diffnorm")


        # Add cross entropy etc
        self.optimizer = tf.train.AdamOptimizer()
        self.mini = self.optimizer.minimize(self.diffsq)

        self.maxerror = tf.reduce_max(tf.abs(self.differencevec))


        # Also categorize mistakes into good (same sign outcome) and bad (different sign outcome)

    def run_step(self, sess, inp, out):
        a = sess.run(self.mini, {self.data:inp, self.target:out})
        b = sess.run(self.diffsq, {self.data:inp, self.target:out})
        return b

    def calc_error(self, sess, test_x, test_y):
        return  sess.run(self.diffsq, {self.data:test_x, self.target:test_y})

def expand_to_one_hot(rows, sds):
    len, ninstr = rows.shape
    newarr = np.array([len, ninstr*6])
    for i in range(len):
        for j in range(ninstr):
            thissd = rows[i, j]/sds[j]

            if (thissd < -2.0): newarr[i,6*j:6*(j+1)] = [1, 0, 0, 0, 0, 0]
            if (thissd >= -2.0) and (thissd < -1.0): newarr[i,6*j:6*(j+1)] = [0, 1, 0, 0, 0, 0]
            if (thissd >= -1.0) and (thissd < 0): newarr[i,6*j:6*(j+1)] = [0, 0, 1, 0, 0, 0]
            if (thissd >= 0) and (thissd < 1.0): newarr[i,6*j:6*(j+1)] = [0, 0, 0, 1, 0, 0]
            if (thissd >= 1.0) and (thissd <= 2.0): newarr[i, 6*j:6*(j+1)] = [0, 0, 0, 0, 1, 0]
            if (thissd > 2.0): newarr[i, 6*j:6*(j+1)] = [0, 0, 0, 0, 1, 0]

    return newarr

if __name__ == '__main__':

    baseDir = r'/media/sachitm/DATA/ML Data'
    x = FileDB(baseDir)

    r4_x = x.readArrayBin('Results', 'r4_x.npy')
    r4_y = x.readArrayBin('Results', 'r4_y.npy')
    r4_t = x.readTime('Results', 'r4_t')


    # Split into a training and validation set
    # Take 50000 to validate and the rest for training

    # Extract data from this
    nrows, ninstr, nts  = r4_x.shape
    validation_rows = np.zeros(nrows, dtype=np.bool)
    validation_rows[sample(range(nrows), 50000)] = True
    test_r4_x = r4_x[validation_rows]
    test_r4_y = r4_y[validation_rows]
    test_r4_t = r4_t[validation_rows]

    # Normalize training data to convert to stds
    train_r4_x = r4_x[np.logical_not(validation_rows)]
    train_r4_y = r4_y[np.logical_not(validation_rows)]
    train_r4_t = r4_t[np.logical_not(validation_rows)]

    x_std   = np.std(train_r4_x, axis=(0,2))

    # Convert x's into std deviations
    train_x_sds = np.divide(train_r4_x.transpose([1, 0, 2]), x_std[:,np.newaxis, np.newaxis]).transpose([1,0,2])
    train_y_sds = np.divide(train_r4_y.transpose(), x_std[:,np.newaxis]).transpose()
    keep = np.max(np.abs(train_x_sds), axis=(1,2))<5
    train_x_sds = train_x_sds[keep]
    train_y_sds = train_y_sds[keep]
    train_r4_t_sds = train_r4_t[keep]

    # Now convert the y's to [-inf, -2], [-2 -1] [-1 0] [0 1] [1 2] [2 inf] 6 classes per for fitting
    test_r4_x_sds = np.divide(test_r4_x.transpose([1, 0, 2]), x_std[:,np.newaxis, np.newaxis]).transpose([1,0,2])
    test_r4_y_sds = np.divide(test_r4_y.transpose(), x_std[:,np.newaxis]).transpose()



    res = LSTMResidual(ninstr=11, seqdepth=12)
    batchsize = 10000

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    for i in range(2000):
        # pick batchsize rows
        rowidx = sample(range(len(train_x_sds)), batchsize)
        insamperr = res.run_step(sess, train_x_sds[rowidx].transpose([0, 2, 1]), train_y_sds[rowidx])
        outsamperr = res.calc_error(sess, train_x_sds.transpose([0, 2, 1]), train_y_sds)

        if (0 == i%1):
            print("Iter  {0:5d}   InSamp  {1:.4f}  OutSamp  {2:.4f}".format(i, insamperr, outsamperr))


    print("stop here")


