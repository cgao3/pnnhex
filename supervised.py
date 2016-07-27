from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import time
import numpy as np

from layer import Layer
from read_data import *
from six.moves import xrange
import os

TRAIN_DATA_PATH="data/7x7rawgames.dat"
TEST_DATA_PATH="data/test7x7.dat"

MODELS_DIR="models/"
SLMODEL_NAME="slmodel.ckpt"

tf.app.flags.DEFINE_boolean("training", True, "False if for evaluation")
tf.app.flags.DEFINE_integer("num_epoch", 100, "number of epoches")
FLAGS = tf.app.flags.FLAGS

class SLNetwork(object):
    '''
    The first layer is input layer, using VALID padding, the rest use SAME padding
    kernal default size 3x3
    kernal depth 80 every layer
    '''
    def __init__(self, trainDataPath=None, testDataPath=None):
        self.train_data_path=trainDataPath
        self.test_data_path=testDataPath

    def declare_layers(self, num_hidden_layer):
        self.num_hidden_layer=num_hidden_layer
        self.input_layer = Layer("input_layer", paddingMethod="VALID")
        self.conv_layer=[None]*num_hidden_layer
        for i in range(num_hidden_layer):
            self.conv_layer[i] = Layer("conv%d_layer" % i)

    #will reue this model for evaluation
    def model(self, data_node, kernal_size=(3,3), kernal_depth=80, value_net=False):
        output = [None] * self.num_hidden_layer
        weightShape0=kernal_size+(INPUT_DEPTH, kernal_depth)
        output[0]=self.input_layer.convolve(data_node, weight_shape=weightShape0, bias_shape=(kernal_depth,))

        weightShape=kernal_size+(kernal_depth,kernal_depth)
        for i in range(self.num_hidden_layer-1):
            output[i+1]=self.conv_layer[i].convolve(output[i], weight_shape=weightShape, bias_shape=(kernal_depth,))

        logits=self.conv_layer[self.num_hidden_layer-1].move_logits(output[self.num_hidden_layer-1], BOARD_SIZE)

        return logits


    def train(self, num_epochs):
        self.train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.train_labels_node = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
        self.eval_data_node = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.eval_label_node = tf.placeholder(tf.int32, shape=(EVAL_BATCH_SIZE,))

        train_data_util=data_util()
        train_data_util.load_offline_data(self.train_data_path, train_data=True)
        test_data_util=data_util()
        test_data_util.load_offline_data(self.test_data_path,train_data=False)
        print("train and test data loaded..")
        self.declare_layers(num_hidden_layer=8)
        logits=self.model(self.train_data_node)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.train_labels_node))
        train_prediction = tf.nn.softmax(logits)
        tf.get_variable_scope().reuse_variables()
        eval_prediction=tf.nn.softmax(self.model(self.eval_data_node))

        #learning_rate_global_step=tf.Variable(0)
        #starting_rate=0.1/BATCH_SIZE
        #train_step=100000 # learning rate *0.95 every train_step feed
        #learning_rate = tf.train.exponential_decay(starting_rate, learning_rate_global_step * BATCH_SIZE, train_step, 0.9, staircase=True)
        #opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=learning_rate_global_step)
        opt2=tf.train.AdamOptimizer().minimize(loss)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init=tf.initialize_all_variables()
            sess.run(init)
            print("Initialized!")
            offset1, offset2 = 0, 0
            nepoch = 0
            step=0
            print_frquence=10
            test_frequence=100
            while (nepoch < num_epochs):
                off1, off2, nextepoch = train_data_util.prepare_batch(offset1, offset2)
                x = train_data_util.batch_states.astype(np.float32)
                y = train_data_util.batch_labels.astype(np.int32)
                feed_diction = {self.train_data_node: x,
                                self.train_labels_node: y}
                _, loss_value, predictions = sess.run([opt2, loss, train_prediction], feed_dict=feed_diction)
                offset1, offset2 = off1, off2
                if (nextepoch):
                    nepoch += 1
                if step % print_frquence == 0:
                    print("epoch:", nepoch, "loss: ", loss_value, "error rate:",
                          error_rate(predictions, train_data_util.batch_labels))
                if step % test_frequence == 0:
                    test_data_util.prepare_batch(0,0)
                    x=test_data_util.batch_states.astype(np.float32)
                    y=test_data_util.batch_labels.astype(np.int32)
                    feed_d={self.eval_data_node:x, self.eval_label_node:y}
                    predict=sess.run(eval_prediction, feed_dict=feed_d)
                    print("evaluation error rate", error_rate(predict, test_data_util.batch_labels))
                step += 1

            sl_model_dir=os.path.dirname(MODELS_DIR)
            if not os.path.exists(sl_model_dir):
                print("creating dir ", sl_model_dir)
                os.makedirs(sl_model_dir)
            saver.save(sess, os.path.join(sl_model_dir, SLMODEL_NAME))

def error_rate(predictions, labels):
    return 100.0 - 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]

def main(argv=None):
    supervisedlearn=SLNetwork(TRAIN_DATA_PATH, TEST_DATA_PATH)
    num_epochs=FLAGS.num_epoch
    supervisedlearn.train(num_epochs)

if __name__ == "__main__":
    tf.app.run()

