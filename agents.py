from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from game_util import *
from unionfind import *
from policygradient import PGNetwork
import threading
from program import Program

class WrapperAgent(object):

    def __init__(self, executable):
        self.executable=executable
        self.program=Program(self.executable, True)
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

    def sendStates(self, intgamestate):
        turn=len(intgamestate)%2
        for m in intgamestate:
            raw_m=intmove_to_raw(m)
            if turn ==0:
                self.play_black(raw_m)
            else: self.play_white(raw_m)
            turn=(turn+1)%2

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

class NNAgent(object):

    def __init__(self, model_location, name):
        self.model_path=model_location
        self.agent_name=name
        self.initialize_game()

    def initialize_game(self):
        self.game_state=[]
        self.boardtensor=np.zeros(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        make_empty_board_tensor(self.boardtensor)
        self.load_model()

    def load_model(self):
        self.data_node = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.sess=tf.Session()
        self.network = PGNetwork("neural net agent")
        self.logit=self.network.model(self.data_node)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def reinitialize(self):
        self.game_state = []
        make_empty_board_tensor(self.boardtensor)

    #0-black player, 1-white player
    def play_move(self, intplayer, intmove):
        update_tensor(self.boardtensor, intplayer, intmove)
        self.game_state.append(intmove)

    def generate_move(self):
        logits=self.sess.run(self.logit, feed_dict={self.data_node:self.boardtensor})
        intmove=softmax_selection(logits, self.game_state)
        #intmove=max_selection(logits, self.game_state)
        raw_move=intmove_to_raw(intmove)
        assert(ord('a') <= ord(raw_move[0]) <= ord('z') and 0<= int(raw_move[1:]) <BOARD_SIZE**2)
        return raw_move

    def close_all(self):
        self.sess.close()
