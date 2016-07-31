from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from game_util import *
from supervised import SLNetwork
import threading
from program import Program

#WrapperAgent can wrap an exe (mohex/wolve) or an exe_nn_agent that implements the GtpInterface
class WrapperAgent(object):

    def __init__(self, executable, verbose=False):
        self.executable=executable
        self.program=Program(self.executable, verbose)
        self.name=self.program.sendCommand("name").strip()
        self.lock=threading.Lock()

    def sendCommand(self, command):
        self.lock.acquire()
        answer=self.program.sendCommand(command)
        self.lock.release()
        return answer

    def reconnect(self):
        self.program.terminate()
        self.program=Program(self.executable, True)
        self.lock=threading.Lock()

    def play_black(self, move):
        self.sendCommand("play black "+move)

    def play_white(self, move):
        self.sendCommand("play white "+move)

    def genmove_black(self):
        return self.sendCommand("genmove black").strip()

    def genmove_white(self):
        return self.sendCommand("genmove white").strip()

    def clear_board(self):
        self.sendCommand("clear_board")

    def set_board_size(self, size):
        self.sendCommand("boardsize "+repr(size))

    def play_move_seq(self, moves_seq):
        turn=0
        for m in moves_seq:
            self.play_black(m) if turn==0 else self.play_white(m)
            turn = (turn+1)%2

class NNAgent(object):

    def __init__(self, model_location, name, is_value_net=False):
        self.model_path=model_location
        self.agent_name=name
        self.is_value_net=is_value_net
        self.initialize_game()

    def initialize_game(self):
        self.game_state=[]
        self.boardtensor=np.zeros(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        make_empty_board_tensor(self.boardtensor)
        self.load_model()

    def load_model(self):
        self.data_node = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.sess=tf.Session()
        if self.is_value_net :
            from valuenet import ValueNet
            self.vnet=ValueNet()
            self.keep_prob_node=tf.placeholder(tf.float32)
            self.value=self.vnet.model(self.data_node, keep_prob_node=self.keep_prob_node)
            self.position_values=np.ndarray(dtype=np.float32, shape=(BOARD_SIZE**2,))
        else:
            self.net = SLNetwork()
            self.net.declare_layers(num_hidden_layer=8)
            self.logit=self.net.model(self.data_node)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def reinitialize(self):
        self.game_state = []
        make_empty_board_tensor(self.boardtensor)

    #0-black player, 1-white player
    def play_move(self, intplayer, intmove):
        update_tensor(self.boardtensor, intplayer, intmove)
        self.game_state.append(intmove)

    def generate_move(self, intplayer=None):
        if self.is_value_net :
            s=list(self.game_state)
            empty_positions=[i for i in range(BOARD_SIZE**2) if i not in s]
            self.position_values.fill(0.0)
            for intmove in empty_positions:
                update_tensor(self.boardtensor, intplayer, intmove)
                v=self.sess.run(self.value, feed_dict={self.data_node:self.boardtensor})
                undo_update_tensor(self.boardtensor,intplayer, intmove)
                self.position_values[intmove]=v
            im=softmax_selection(self.position_values, self.game_state, temperature=0.1)
            #im=max_selection(self.position_values, self.game_state)
            return im
        else:
            logits=self.sess.run(self.logit, feed_dict={self.data_node:self.boardtensor})
            intmove=softmax_selection(logits, self.game_state)
            #intmove=max_selection(logits, self.game_state)
            raw_move=intmove_to_raw(intmove)
            assert(ord('a') <= ord(raw_move[0]) <= ord('z') and 0<= int(raw_move[1:]) <BOARD_SIZE**2)
            return raw_move

    def close_all(self):
        self.sess.close()
