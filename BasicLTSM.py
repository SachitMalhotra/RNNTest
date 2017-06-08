import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from random import shuffle
from random import sample

"""
Given a string of zeros and one count the number of ones in it
"""
class BasicLTSM(object):
    NUM_EXAMPLES = 100000

    def make_training_set(self):
       train_input = [i for i in range(2**20)]
       shuffle(train_input)
       train_input = np.reshape(train_input, [len(train_input), 1])
       train_input = np.unpackbits(train_input.view(np.uint8), axis=1)
       train_input = train_input[:,:20]
       output = np.sum(train_input, axis=1)

       yy = np.zeros([len(output), 21])
       for i, idx in enumerate(output):
           yy[i, idx] = 1.0

       return train_input, yy


    def __init__(self):

        train_input, train_output = self.make_training_set()
        self.test_input = train_input[BasicLTSM.NUM_EXAMPLES:]
        self.test_output = train_output[BasicLTSM.NUM_EXAMPLES:]

        self.train_input = train_input[:BasicLTSM.NUM_EXAMPLES]
        self.train_output = train_output[:BasicLTSM.NUM_EXAMPLES]

        # batch size x seq_length x input_dimension
        self.data    = tf.placeholder(tf.float32, [None, 20, 1], name="Data")
        self.target  = tf.placeholder(tf.float32, [None, 21], name="Target")

        num_hidden = 24
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)
        self.val, self.state = tf.nn.dynamic_rnn(self.cell, self.data, dtype=tf.float32 )
        self.val = tf.transpose(self.val, [1, 0, 2])
        self.last = tf.gather(self.val, int(self.val.get_shape()[0])-1)

        self.weight = tf.Variable(tf.truncated_normal([num_hidden, int(self.target.get_shape()[1])]))
        self.bias   = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))

        self.prediction = tf.matmul(self.last, self.weight) + self.bias
        self.xent      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.target))

        # Add cross entropy etc
        self.optimizer = tf.train.AdamOptimizer()
        self.mini = self.optimizer.minimize(self.xent)

        self.mistakes = tf.not_equal(tf.argmax(self.prediction, 1), tf.argmax(self.target, 1))
        self.errorrate = tf.reduce_mean(tf.cast(self.mistakes, dtype=tf.float32))

    def run_step(self, sess, inp, out):
        a = sess.run(self.mini, {self.data:inp, self.target:out})
        b = sess.run(self.xent, {self.data:inp, self.target:out})
        return b

    def calc_error(self, sess, test_x, test_y):
        return  sess.run(self.errorrate, {self.data:test_x, self.target:test_y})




if __name__ == '__main__':

    # Instantiate variable
    x = BasicLTSM()

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    batch_size = 1000
    ntest = BasicLTSM.NUM_EXAMPLES
    test_x = np.reshape(x.test_input, [len(x.test_input), 20, 1]).astype(np.float32)
    test_y = x.test_output.astype(np.float32)

    for i in range(10000):
        idxs = sample(range(ntest), batch_size)
        this_x = np.reshape(x.train_input[idxs].astype(np.float32), [batch_size, 20, 1])
        this_y = x.train_output[idxs].astype(np.float32)

        in_samp_error = x.run_step(sess,  this_x, this_y)
        out_samp_error = x.calc_error(sess, test_x, test_y)

        if (0 == i%1):
            print("Iter {0:5d} Xent  {1:.5f}  Error {2:.5f}  ".format(i, in_samp_error, out_samp_error))


    sess.close()






