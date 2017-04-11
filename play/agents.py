from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from utils.game_util import *
from neuralnet.supervised import SupervisedNet
import threading
from play.program import Program
from utils.commons import *

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
        self.tensorUtil=RLTensorUtil13x13()
        self.model_path=model_location
        self.agent_name=name
        self.is_value_net=is_value_net

        self.initialize_game([])

    def initialize_game(self, initialRawMoveList):
        self.game_state=[]
        for rawmove in initialRawMoveList:
            self.game_state.append(MoveConvertUtil.rawMoveToIntMove(rawmove))

        self.boardtensor=np.zeros(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))

        self.tensorUtil.set_position_tensors_in_batch(self.boardtensor,0,self.game_state)
        self.load_model()

    def load_model(self):
        self.data_node = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.sess=tf.Session()
        if self.is_value_net :
            from neuralnet.valuenet import ValueNet2
            self.vnet=ValueNet2()
            self.value=self.vnet.model(self.data_node)
            self.position_values=np.ndarray(dtype=np.float32, shape=(BOARD_SIZE**2,))
        else:
            self.net = SupervisedNet(srcTestDataPath=None,srcTrainDataPath=None, srcTestPathFinal=None)
            self.net.setup_architecture(nLayers=7)
            self.logit=self.net.model(self.data_node)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def reinitialize(self, moveList=None):
        self.game_state = []
        if(moveList):
            for rawmove in moveList:
                self.game_state.append(MoveConvertUtil.rawMoveToIntMove(rawmove))
        self.boardtensor.fill(0)
        self.tensorUtil.set_position_tensors_in_batch(self.boardtensor, 0, self.game_state)

    #0-black player, 1-white player
    def play_move(self, intplayer, intmove):
        self.game_state.append(intmove)
        self.boardtensor.fill(0)
        self.tensorUtil.set_position_tensors_in_batch(self.boardtensor, 0, self.game_state)

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
            rawMove=MoveConvertUtil.intMoveToRaw(intMove=intmove)
            assert(ord('a') <= ord(rawMove[0]) <= ord('z') and 0<= int(rawMove[1:]) <BOARD_SIZE**2)
            return rawMove

    def close_all(self):
        self.sess.close()
