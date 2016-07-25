from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from six.moves import xrange

import numpy as np

import tensorflow as tf

from unionfind import unionfind
from game_util import *
from layer import Layer
from agents import WrapperAgent
import os
import sys

EXAMPLES_PATH = "vexamples/"+repr(BOARD_SIZE)+"x"+repr(BOARD_SIZE)+"examples.dat"

class RandomPlayer(object):
    def __init__(self):
        pass
    @staticmethod
    def uniform_random_genmove(intgamestate):
        N = BOARD_SIZE ** 2
        empty_positions = [i for i in range(N) if i not in intgamestate]
        return np.random.choice(empty_positions)


class ExampleProducer(object):
    def __init__(self, exe_path1, exe_path2, num_examples):
        self.num_examples=num_examples
        self.set_players(exe_path1, exe_path2)

    def set_players(self, exe_path1, exe_path2):
        self.fast_player = WrapperAgent(exe_path1)
        self.strong_player = WrapperAgent(exe_path2)

    def produce_data(self):
        if not os.path.exists(os.path.dirname(EXAMPLES_PATH)):
            os.makedirs(os.path.dirname(EXAMPLES_PATH))
        count = 0
        fout = open(EXAMPLES_PATH, "w+")
        while count < self.num_examples:
            example = self.generate_one_example(True, 0.5, 1.0)
            if example:
                raw_game, label=example
                count += 1
                for m in raw_game:
                    fout.write(m + " ")
                fout.write(repr(label) + "\n")
        fout.close()

    def generate_one_example(self, indicate_boardsize=False, time_limit1=None, time_limit2=None):
        if indicate_boardsize:
            self.fast_player.set_board_size(BOARD_SIZE)
            self.strong_player.set_board_size(BOARD_SIZE)

        if time_limit1:
            self.fast_player.sendCommand("param_wolve max_time "+ repr(time_limit1))

        if time_limit2:
            self.strong_player.sendCommand("param_wolve max_time "+ repr(time_limit2))

        U = np.random.randint(0, BOARD_SIZE ** 2)
        g = []
        black_groups = unionfind()
        white_groups = unionfind()
        turn = 0
        status = -1
        # Fast Player play to U
        for i in range(0, U):
            move = self.fast_player.genmove_black() if turn == 0 else self.fast_player.genmove_white()
            if move == "resign":
                return False
            intmove = raw_move_to_int(move)
            black_groups, white_groups = update_unionfind(intmove, turn, g, black_groups, white_groups)
            status = winner(black_groups, white_groups)
            turn = (turn + 1) % 2
            g.append(intmove)
            if (status == 0 or status == 1): return False

        # Random play at step U
        int_random_move = RandomPlayer.uniform_random_genmove(g)
        black_groups, white_groups = update_unionfind(int_random_move, turn, g, black_groups, white_groups)
        status = winner(black_groups, white_groups)
        g.append(int_random_move)
        example_state = list(g)
        example_state_player = turn
        turn = (turn + 1) % 2
        if status == 0 or status == 1: return False

        # Strong play from U+1 till game ends
        while status == -1:
            move = self.strong_player.genmove_black() if turn == 0 else self.strong_player.genmove_white()
            if move == "resign":
                status = 0 if turn ==1 else 1
                break
            intmove = raw_move_to_int(move)
            black_groups, white_groups = update_unionfind(intmove, turn, g, black_groups, white_groups)
            status = winner(black_groups, white_groups)
            turn = (turn + 1) % 2
            g.append(intmove)
        R = 1.0 if status == example_state_player else -1.0

        return self.build_example(example_state), R

    # label is win or loss, +1 or -1, OR modified game reward
    def build_example(self, intgamestate):
        raw_state = []
        for m in intgamestate:
            rawmove = intmove_to_raw(m)
            raw_state.append(rawmove)
        return (raw_state)

def main(argv=None):
    exe1 = "/home/cgao3/benzene/src/wolve/wolve 2>/dev/null"
    exe2 = "/home/cgao3/benzene/src/wolve/wolve 2>/dev/null"
    eproducer=ExampleProducer(exe1, exe2, 100000)
    eproducer.produce_data()

if __name__ == "__main__":
    tf.app.run()