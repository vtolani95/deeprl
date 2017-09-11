import numpy as np
import tensorflow as tf

ENV_NAME = 'Reacher-v1'
VERSION = '1.0'
NUM_ACTIONS = 2
NUM_OBS = 11
FC1_SIZE = 10
FC2_SIZE = 4

weights = {'fc1': tf.get_variable('fc1', 
            shape=[NUM_OBS, FC1_SIZE],
            initializer=tf.contrib.layers.xavier_initializer()),
            'fc2': tf.get_variable('fc2',
            shape=[FC1_SIZE, FC2_SIZE],
            initializer=tf.contrib.layers.xavier_initializer()),
            'fc3': tf.get_variable('fc3',
            shape=[FC2_SIZE, NUM_ACTIONS],
            initializer=tf.contrib.layers.xavier_initializer())}

biases = {'b1': tf.get_variable("b1",
                shape=[FC1_SIZE],
                initializer=tf.constant_initializer(0.0)),
        'b2': tf.get_variable("b2",
                shape=[FC2_SIZE],
                initializer=tf.constant_initializer(0.0)),
        'b3': tf.get_variable("b3",
                shape=[NUM_ACTIONS],
                initializer=tf.constant_initializer(0.0))}

def model(input_layer, keep_prob):
    fc1 = tf.nn.relu(tf.matmul(input_layer, weights['fc1']+biases['b1']))
    fc2 = tf.nn.relu(tf.matmul(fc1, weights['fc2']+biases['b2']))
    dropout = tf.nn.dropout(fc2, keep_prob)
    preds = tf.matmul(dropout, weights['fc3'] + biases['b3'])
    return preds    


