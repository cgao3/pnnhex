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
from game_util import *

VALUE_NET_BATCH_SIZE=64

NUM_EXAMPLES=1e6

class ValueNet(object):

    def __init__(self):
      self.games=None
      self.batch_states=np.ndarray(dtype=np.float32, shape=(VALUE_NET_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
      self.batch_label=np.ndarray(dtype=np.float32, shape=(VALUE_NET_BATCH_SIZE,))
      pass

    def regression(self):
        pass

    def read_examples(self, data_path):
        if self.games==None:
            self.games=[]
            with open(data_path, "r") as f:
                for line in f:
                    self.games.append(line.split())
        num_epochs=10
        epoch_count=0
        offset=0
        while epoch_count < num_epochs:
            offset, next_epoch=self.prepare_batch(offset, batchsize=VALUE_NET_BATCH_SIZE)

            self.regression()

            if next_epoch:
                epoch_count += 1

    def make_empgy(self, kth):
        self.batch_states[kth].fill(0)
        # black occupied
        self.batch_states[kth][0, 0:INPUT_WIDTH, 0, 0] = 1
        self.batch_states[kth][0, 0:INPUT_WIDTH, INPUT_WIDTH - 1, 0] = 1
        # white occupied
        self.batch_states[kth][0, 0, 1:INPUT_WIDTH - 1, 1] = 1
        self.batch_states[kth][0, INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 1] = 1
        # empty positions
        self.batch_states[kth][0, 1:INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, INPUT_DEPTH - 1] = 1

    def build_single_example(self, row_i, kth):
        R=self.games[row_i][-1]
        self.make_empgy(kth)
        turn=0
        for j in range(len(self.games[row_i])-1):
            move=self.games[row_i][j]
            x,y=raw_move_to_pair(move)
            self.batch_states[kth, x+1, y+1, turn]=1
            self.batch_states[kth, x+1, y+1, 2]=0
            turn = (turn+1)%2

    def prepare_batch(self, offset, batchsize):
        count=0
        new_offset=offset
        next_epoch=False
        while count<batchsize:
            for i in xrange(offset, len(self.games)):
                self.build_single_example(i, count)
                count += 1
                if count >= batchsize:
                    new_offset=i+1
                    break
            if count < batchsize:
                offset=0
                next_epoch=True
        return new_offset, next_epoch

    # the same structure as supervised network
    def model(self, data_node):
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