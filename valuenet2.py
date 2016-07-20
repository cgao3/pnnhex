from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from six.moves import xrange

import numpy as np

import tensorflow as tf

from unionfind import unionfind
from game_util import  *
from layer import Layer
from agents import WrapperAgent

VALUE_NET_BATCH_SIZE=64

NUM_EXAMPLES=1e6

class ValueNet(object):

    def __init__(self):
      self.games=None
      self.batch_states=np.ndarray(dtype=np.float32, shape=(VALUE_NET_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
      self.batch_label=np.ndarray(dtype=np.float32, shape=(VALUE_NET_BATCH_SIZE))
      pass

    def regression(self):
        pass
    def read_examples(self, data_path):
        if self.games==None:
            self.games=[]
            with open(data_path, "r") as f:
                for line in f:
                    self.games.append(line.split())

    def build_single_example(self, offset, kth):
        R=self.games[offset][-1]
        

    def prepare_batch(self, offset, batchsize):
        count=0
        while count<batchsize:
            for i in xrange(offset, len(self.games)):


    # the same structure as supervised network
    def model(self):
        data_node=tf.placeholder(dtype=tf.float32, shape=(VALUE_NET_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        num_hidden_layer=8
        self.num_hidden_layer = num_hidden_layer
        self.input_layer = Layer("input_layer", paddingMethod="VALID")
        self.conv_layer = [None] * num_hidden_layer

        for i in range(num_hidden_layer):
            self.conv_layer[i] = Layer("conv%d_layer" % i)

        kernal_size=(3, 3)
        kernal_depth=80

        output = [None] * self.num_hidden_layer
        weightShape0 = kernal_size + (INPUT_DEPTH, kernal_depth)
        output[0] = self.input_layer.convolve(data_node, weight_shape=weightShape0, bias_shape=(kernal_depth,))

        weightShape = kernal_size + (kernal_depth, kernal_depth)
        for i in range(self.num_hidden_layer - 1):
            output[i + 1] = self.conv_layer[i].convolve(output[i], weight_shape=weightShape,
                                                        bias_shape=(kernal_depth,))
        logits = self.conv_layer[self.num_hidden_layer - 1].one_filter_out(output[self.num_hidden_layer - 1],
                                                                               BOARD_SIZE)

        return logits

def main(argv=None):
    exe1="/home/cgao3/benzene/src/wolve/wolve 2>/dev/null"
    exe2="/home/cgao3/benzene/src/mohex/mohex 2>/dev/null"
    vnet=ValueNet()

if __name__ == "__main__":
    tf.app.run()