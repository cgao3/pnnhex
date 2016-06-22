
from read_data import BOARD_SIZE, INPUT_WIDTH, INPUT_DEPTH

import numpy as np
from unionfind import unionfind
import tensorflow as tf
from supervised import SLNetwork
from layer import *

PG_BATCH_SIZE=128
batch_games=[]

black=unionfind()
white=unionfind()
class PGNetwork(object):

    def __init__(self, modelpath):
        self.model_location=modelpath
        self.config(num_hidden_layer=8)

    #the same structure as supervised network
    def config(self, num_hidden_layer, kernal_size=(3, 3), kernal_depth=80):
        self.input_data_node = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))

        self.input_layer = Layer("input_layer", paddingMethod="VALID")
        self.conv_layer = [None] * num_hidden_layer
        self.output = [None] * num_hidden_layer
        weightShape0 = kernal_size + (INPUT_DEPTH, kernal_depth)
        self.output[0] = self.input_layer.convolve(self.train_data_node, weight_shape=weightShape0,
                                                   bias_shape=(kernal_depth,))
        for i in range(num_hidden_layer):
            self.conv_layer[i] = Layer("conv%d_layer" % i)

        weightShape = kernal_size + (kernal_depth, kernal_depth)
        for i in range(num_hidden_layer - 1):
            self.output[i + 1] = self.conv_layer[i].convolve(self.output[i], weight_shape=weightShape,
                                                             bias_shape=(kernal_depth,))
        self.logits = self.conv_layer[num_hidden_layer - 1].one_filter_out(self.output[num_hidden_layer - 1], BOARD_SIZE)

    #input is raw score such as [-20,30,10]
    def softmax_selection(self, logits, currentstate):
        empty_positions=[i for i in range(BOARD_SIZE**2) if i not in currentstate]
        logits=np.squeeze(logits)
        print(empty_positions)
        print(logits)
        effective_logits=[logits[i] for i in empty_positions]
        max_value=np.max(effective_logits)
        effective_logits=effective_logits - max_value
        effective_logits=np.exp(effective_logits)
        sum_value=np.sum(effective_logits)
        prob=effective_logits/sum_value
        v=np.random.random()
        sum_v=0.0
        action=None
        for i, e in enumerate(prob):
            sum_v += e
            if(sum_v >= v):
                action=i
                break
        return empty_positions[action]


    def forwardpass(self, input_tensor, currentboard):
        saver=tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, self.model_location)
            logit=sess.run([self.logits], feed_dict={self.input_data_node:input_tensor})
            return self.softmax_selection(logit,currentboard)

    def selfplay(self, other_player):
        pass

    def prepare_batchgames(self):
        pass

if __name__ == "__main__":
    pgtest = PGNetwork("hello")
    logit=np.random.rand(1,5)

    moves=[1,3]
    a=pgtest.softmax_selection(logit, moves)
    print("action selected:", a)