import numpy as np
import tensorflow as tf
from random import shuffle

"""
Given a string of zeros and one count the number of ones in it
"""
class BasicLTSM(object):
    NUM_EXAMPLES = 100000

    @staticmethod
    def make_training_set():
       train_input = ['{0:020b}'.format(i) for i in range(2**20)]
       shuffle(train_input)
       train_input = [map(int, i) for i in train_input]
       ti = []
       for i in train_input:
           temp_list = []
           for j in i:
               temp_list.append(j)
           ti.append(np.array(temp_list))
       train_input = ti

       train_output = []
       for i in train_input:
           count = 0
           for j in i:
               if (j[0] == 1): count += 1
           temp_list = ([0] * 21)
           temp_list[count] = 1
           train_output.append(temp_list)

       return train_input, train_output

    train_input, train_output = make_training_set()
    test_input  = train_input[NUM_EXAMPLES:]
    test_output = train_output[NUM_EXAMPLES:]

    train_input = train_input[:NUM_EXAMPLES]
    train_output= train_output[:NUM_EXAMPLES]

    def __init__(self):
        # batch size x seq_length x input_dimension
        self.data    = tf.placeholder(tf.float32, [None, 20, 1])
        self.target  = tf.placeholder(tf.float32, [None, 21])

        num_hidden = 24
        self.cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        self.val, self.state = tf.nn.dynamic_rnn(self.cell, self.data, dtype=tf.float32 )
        self.val = tf.transpose(self.val, [1, 0, 2])
        self.last = tf.gather(self.val, int(self.val.get_shape()[0])-1)

        self.weight = tf.Variable(tf.truncated_normal([num_hidden, int(self.target.get_shape()[1])]))
        self.bias   = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))

        self.prediction = tf.nn.softmax(tf.matmul(self.last, self.weight) + self.bias)

        # Add cross entropy etc





