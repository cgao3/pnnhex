from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from layer import Layer
from read_data import *
from six.moves import xrange

tf.app.flags.DEFINE_boolean("training", True, "False if for evaluation")
tf.app.flags.DEFINE_string("check_point_dir", "savedModel/", "path to save the model")
tf.app.flags.DEFINE_string("train_data_dir", "data/", "path to training data")
tf.app.flags.DEFINE_string("test_data_dir", "test/", "path to test data")
FLAGS = tf.app.flags.FLAGS

class SLNetwork(object):
    '''
    The first layer is input layer, using VALID padding, the rest use SAME padding
    kernal default size 3x3
    kernal depth 80 every layer
    '''
    def __init__(self, trainDataPath, testDataPath):
        self.train_data_path=trainDataPath
        self.test_data_path=testDataPath

    def config(self, num_hidden_layer, kernal_size=(3,3), kernal_depth=80):
        self.train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.train_labels_node = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
        self.input_layer = Layer("input_layer", paddingMethod="VALID")

        self.conv_layer=[None]*num_hidden_layer
        self.output=[None]*num_hidden_layer
        weightShape0=kernal_size+(INPUT_DEPTH, kernal_depth)
        self.output[0]=self.input_layer.convolve(self.train_data_node, weight_shape=weightShape0, bias_shape=(kernal_depth,))
        for i in range(num_hidden_layer):
            self.conv_layer[i]=Layer("conv%d_layer"%i)

        weightShape=kernal_size+(kernal_depth,kernal_depth)
        for i in range(num_hidden_layer-1):
            self.output[i+1]=self.conv_layer[i].convolve(self.output[i], weight_shape=weightShape, bias_shape=(kernal_depth,))

        self.logits=self.conv_layer[num_hidden_layer-1].one_filter_out(self.output[num_hidden_layer-1], BOARD_SIZE)

    #call config before train..
    def train(self, num_epochs):
        read_raw_data(self.train_data_path)
        print("data loaded..")

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.train_labels_node))
        train_prediction = tf.nn.softmax(self.logits)
        learning_rate_global_step=tf.Variable(0)
        starting_rate=0.01
        train_step=100000 # learning rate *0.95 every train_step feed
        learning_rate = tf.train.exponential_decay(starting_rate, learning_rate_global_step * BATCH_SIZE, train_step, 0.9, staircase=True)

        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=learning_rate_global_step)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init=tf.initialize_all_variables()
            sess.run(init)
            print("Initialized!")
            offset1, offset2 = 0, 0
            nepoch = 0
            step=0
            print_frquence=10
            while (nepoch < num_epochs):
                off1, off2, nextepoch = prepare_batch(offset1, offset2)
                x = batch_states.astype(np.float32)
                y = batch_labels.astype(np.int32)
                feed_diction = {self.train_data_node: x,
                                self.train_labels_node: y}
                _, loss_value, predictions = sess.run([opt, loss, train_prediction], feed_dict=feed_diction)
                offset1, offset2 = off1, off2
                if (nextepoch):
                    nepoch += 1
                if step % print_frquence == 0:
                    print("epoch:", nepoch, "loss: ", loss_value, "error rate:", error_rate(predictions, batch_labels))
                step += 1

            saver.save(sess, FLAGS.check_point_dir + "/model.ckpt")


def error_rate(predictions, labels):
    return 100.0 - 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]


def main(argv=None):
    supervisedlearn=SLNetwork("data/train_games.dat","test_games.dat")
    supervisedlearn.config(num_hidden_layer=8, kernal_size=(3,3), kernal_depth=80)

    num_epochs=40
    supervisedlearn.train(num_epochs)

if __name__ == "__main__":
    tf.app.run()

