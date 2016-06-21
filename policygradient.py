
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

    def __init__(self, modelpath, input_tensor):
        self.model_location=modelpath

    #the same structure as supervised network
    def config(self, num_hidden_layer, kernal_size=(3, 3), kernal_depth=80):
        self.input_data_node = tf.placeholder(tf.float32, shape=(PG_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))

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

        self.logits = self.conv_layer[num_hidden_layer - 1].one_filter_out(self.output[num_hidden_layer - 1],
                                                                           BOARD_SIZE)

    def forwardpass(self, input_tensor):

        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_location)
            sess.run([opt,train_prediction], feed_dict={})
        return

    def prepare_batchgames(self):
        pass



'''a game state is a sequence of moves'''


def forward_pass(network_input):
    d=[]
    return d

def softmax(prob):
    pass

def init_gamestate():
    return []

def empty_board_tensor():
    tensor=np.zeros(shape=(INPUT_WIDTH,INPUT_WIDTH,INPUT_DEPTH),dtype=np.uint8)
    tensor[0:INPUT_WIDTH,0,0]=1
    tensor[0:INPUT_WIDTH,INPUT_WIDTH-1,0]=1

    tensor[0,1:INPUT_WIDTH-1,1]=1
    tensor[INPUT_WIDTH-1,1:INPUT_WIDTH-1,1]=1

    tensor[1:INPUT_WIDTH-1,1:INPUT_WIDTH-1,2]=1

    return tensor

def gamestate_to_input_tensor(g):
    input_tensor=empty_board_tensor()

    for i,intmove in enumerate(g):
        x=intmove//BOARD_SIZE
        y=intmove%BOARD_SIZE
        input_tensor[x+1,y+1,i%2]=1
        input_tensor[x+1,y+1,2]=0

    return input_tensor

def simulation():

    return

def self_play_simulation():

    return

def prepare_pg_batch():
    g=init_gamestate()
    input_tensor=gamestate_to_input_tensor(g)
    k=0

    while k < PG_BATCH_SIZE:
        move_seq=simulation()
        batch_games.append(move_seq)

        k += 1
        pass




