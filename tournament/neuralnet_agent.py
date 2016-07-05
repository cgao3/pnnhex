from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from game_util import *
from unionfind import *
from policygradient import PGNetwork

class network_agent(object):

    def __init__(self, model_location, name):
        self.model_path=model_location
        self.agent_name=name
        self.initialize_game()

    def initialize_game(self):
        self.game_state=[]
        self.tensor=np.ndarray(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        make_empty_board_tensor(self.tensor)
        self.black_groups=unionfind()
        self.white_groups=unionfind()

    def load_model(self):
        self.data_node = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        saver = tf.train.Saver()
        self.sess=tf.Session()
        self.network = PGNetwork()
        self.logit=self.network.model(self.data_node)
        saver.restore(self.sess, self.model_path)

    #0-black player, 1-white player
    def play_move(self, intplayer, intmove):
        self.game_state.append(intmove)
        update_tensor(self.tensor, intplayer, intmove)
        self.black_groups, self.white_groups=\
            update_unionfind(intmove, intplayer, self.game_state, self.black_groups, self.white_groups)

    def generate_move(self):
        logits=self.sess.run(self.logit, feed_dict={self.data_node:self.tensor})
        intmove=softmax_selection(logits, self.game_state)
        raw_move=intmove_to_raw(intmove)
        assert(ord('a') <= ord(raw_move[0]) <= ord('z') and 0<= int(raw_move[1:]) <BOARD_SIZE**2)
        return raw_move

    def game_status(self):
        return winner(self.black_groups, self.white_groups)

    def close_all(self):
        self.sess.close()


