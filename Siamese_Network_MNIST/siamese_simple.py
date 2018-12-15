'''
Siamese Network Implementation Practice
Lei Mao
10/13/2017
University of Chicago
'''

'''
References
TesorFlow Sharing Variables
https://www.tensorflow.org/versions/r0.12/how_tos/variable_scope/
Simple Siamese Network
https://github.com/ywpkwon/siamese_tf_mnist/blob/master/inference.py
'''
import tensorflow as tf
import os
import keras

LEARNING_RATE = 0.01
SAVE_PERIOD = 500
MODEL_DIR = 'model/'  # path for saving the model
MODEL_NAME = 'siamese_model'
RAND_SEED = 0  # random seed
tf.set_random_seed(RAND_SEED)


class Siamese(object):

    def __init__(self):

        self.input_1 = tf.placeholder(tf.float32, [None, 784], name='input_1')
        self.input_2 = tf.placeholder(tf.float32, [None, 784], name='input_2')
        # self.input_1 = keras.Input(shape=[None, 784], name='input_1')
        # self.input_2 = keras.Input(shape=[None, 784], name='input_2')
        # 1: paired, 0: unpaired
        self.tf_label = tf.placeholder(tf.float32, [None, ], name='label')
        self.output_1, self.output_2 = self.network_initializer()
        self.loss = self.loss_contrastive()
        self.optimizer = self.optimizer_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # Old-school way to initialize fc parameters
    def fc_layer(self, tf_input, n_hidden_units, variable_name):
        # tf_input: batch_size x n_features
        # n_hidden_units: number of hidden units
        assert len(tf_input.get_shape()) == 2
        n_features = tf_input.get_shape()[1]
        tf_weight_initializer = tf.truncated_normal_initializer(mean=0, stddev=0.01)
        # Similar to tf.random_normal_initializer except for the truncation
        tf_bias_initializer = tf.constant_initializer(0.01)
        W = tf.get_variable(
            name=variable_name + '_W',
            dtype=tf.float32,
            shape=[n_features, n_hidden_units],
            initializer=tf_weight_initializer
        )
        b = tf.get_variable(
            name=variable_name + '_b',
            dtype=tf.float32,
            shape=[n_hidden_units],
            initializer=tf_bias_initializer
        )
        fc = tf.nn.bias_add(tf.matmul(tf_input, W), b)
        return fc

    def network(self, input):

        reshaped = tf.reshape(input, shape=[-1, 28, 28, 1])

        w1 = tf.get_variable(shape=[5, 5, 1, 32], dtype=tf.float32, name='w1',
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        b1 = tf.get_variable(shape=[32], dtype=tf.float32, name='b1',
                             initializer=tf.constant_initializer(0.01))

        w2 = tf.get_variable(shape=[3, 3, 32, 64], dtype=tf.float32, name='w2',
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        b2 = tf.get_variable(shape=[64], dtype=tf.float32, name='b2',
                             initializer=tf.constant_initializer(0.01))

        w3 = tf.get_variable(shape=[3, 3, 64, 96], dtype=tf.float32, name='w3',
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        b3 = tf.get_variable(shape=[96], dtype=tf.float32, name='b3',
                             initializer=tf.constant_initializer(0.01))

        conv1 = tf.nn.relu(tf.nn.conv2d(reshaped, w1, strides=[1, 1, 1, 1], padding='VALID') + b1)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='VALID') + b2)

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3 = tf.nn.relu(tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding='VALID') + b3)

        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        flatten = tf.reshape(pool3, [-1, pool3.get_shape()[1]*pool3.get_shape()[2]*96])

        fc1 = self.fc_layer(tf_input=flatten, n_hidden_units=1024, variable_name='fc1')
        ac1 = tf.nn.relu(fc1)

        fc2 = self.fc_layer(tf_input=ac1, n_hidden_units=512, variable_name='fc2')
        ac2 = tf.nn.relu(fc2)

        fc3 = self.fc_layer(tf_input=ac2, n_hidden_units=2, variable_name='fc3')
        # ac3 = tf.nn.tanh(fc3)

        return fc3

        # fc1 = self.fc_layer(tf_input=input, n_hidden_units=1024, variable_name='fc1')
        # ac1 = tf.nn.relu(fc1)
        # fc2 = self.fc_layer(tf_input=ac1, n_hidden_units=1024, variable_name='fc2')
        # ac2 = tf.nn.relu(fc2)
        # fc3 = self.fc_layer(tf_input=ac2, n_hidden_units=2, variable_name='fc3')
        # return fc3

    def network_initializer(self):
        # Initialze neural network
        with tf.variable_scope("siamese") as scope:
            output_1 = self.network(self.input_1)
            scope.reuse_variables()
            output_2 = self.network(self.input_2)
        return output_1, output_2

    def loss_contrastive(self, margin=5.0):
        # Define loss function
        with tf.variable_scope("loss_function") as scope:
            labels = self.tf_label
            # Euclidean distance squared
            eucd2 = tf.pow(tf.subtract(self.output_1, self.output_2), 2, name='eucd2')
            eucd2 = tf.reduce_sum(eucd2, 1)
            # Euclidean distance
            # We add a small value 1e-6 to increase the stability of calculating the gradients for sqrt
            # See https://github.com/tensorflow/tensorflow/issues/4914
            eucd = tf.sqrt(eucd2 + 1e-6, name='eucd')
            # Loss function
            loss_pos = tf.multiply(labels, eucd2, name='constrastive_loss_1')
            loss_neg = tf.multiply(tf.subtract(1.0, labels), tf.pow(tf.maximum(tf.subtract(margin, eucd), 0), 2),
                                   name='constrastive_loss_2')
            loss = tf.reduce_mean(tf.add(loss_neg, loss_pos), name='constrastive_loss')
        return loss

    def optimizer_initializer(self):
        # Initialize optimizer
        # AdamOptimizer and GradientDescentOptimizer has different effect on the final results
        # GradientDescentOptimizer is probably better than AdamOptimizer in Siamese Network
        # optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)
        return optimizer

    def train_model(self, input_1, input_2, label):
        # Train the network
        _, train_loss = self.sess.run([self.optimizer, self.loss],
                                      feed_dict={self.input_1: input_1, self.input_2: input_2,
                                                 self.tf_label: label})
        return train_loss

    def test_model(self, input_1):
        # Test the trained model
        output = self.sess.run(self.output_1, feed_dict={self.input_1: input_1})
        return output

    def load_model(self):
        # Restore the trained model
        assert os.path.exists(MODEL_DIR + MODEL_NAME)
        self.saver.restore(self.sess, MODEL_DIR + MODEL_NAME)

    def save_model(self):
        # Save model routinely
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        # Save the latest trained models
        self.saver.save(self.sess, MODEL_DIR + MODEL_NAME)
