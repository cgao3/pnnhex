from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from six.moves import xrange

import numpy as np

import tensorflow as tf

from unionfind import unionfind
from game_util import *
from agents import WrapperAgent
from layer import Layer
from agents import WrapperAgent
import os
import sys

EXAMPLES_PATH = "vexamples/"+repr(BOARD_SIZE)+"x"+repr(BOARD_SIZE)+"examples.dat"

tf.app.flags.DEFINE_string("exe1_path", "/home/cgao3/benzene/src/wolve/wolve", "exe for the fast player")
tf.app.flags.DEFINE_string("exe2_path", "/home/cgao3/benzene/src/wolve/wolve", "exe for the strong player")
tf.app.flags.DEFINE_string("output_dir","vexamples/"+repr(BOARD_SIZE)+"x"+repr(BOARD_SIZE)+"examples.dat","where to store the examples")
tf.app.flags.DEFINE_integer("num", 10000, "num of examples to produce")
FLAGS=tf.app.flags.FLAGS

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

    def set_players(self, exe_path1, exe_path2, verbose=False):
        self.fast_player = WrapperAgent(exe_path1, verbose)
        self.strong_player = WrapperAgent(exe_path2, verbose)

    def produce_data(self, solving=False):
        if not os.path.exists(os.path.dirname(EXAMPLES_PATH)):
            os.makedirs(os.path.dirname(EXAMPLES_PATH))
        count = 0
        fout = open(EXAMPLES_PATH, "w+")
        while count < self.num_examples:
            if solving:
                example=self.generate_one_example_by_solving()
            else:
                example = self.generate_one_example(True, 0.5, 1.0)
            if example:
                raw_game, label=example
                count += 1
                for m in raw_game:
                    fout.write(m + " ")
                fout.write(repr(label) + "\n")
                print("count==", count)
        fout.close()

    #fast player should be NN
    def generate_one_example_by_solving(self):
        self.fast_player.clear_board()
        self.strong_player.set_board_size(BOARD_SIZE)
        U = np.random.randint(0, BOARD_SIZE ** 2 - BOARD_SIZE)
        g = []
        move_seq = []
        black_groups = unionfind()
        white_groups = unionfind()
        turn = 0
        # Fast Player play to U
        for i in range(U):
            move = self.fast_player.genmove_black() if turn == 0 else self.fast_player.genmove_white()
            if move == "resign":
                return False
            intmove = raw_move_to_int(move)
            black_groups, white_groups = update_unionfind(intmove, turn, g, black_groups, white_groups)
            status = winner(black_groups, white_groups)
            turn = (turn + 1) % 2
            g.append(intmove)
            move_seq.append(move)
            if (status == 0 or status == 1): return False

        # Random play at step U
        int_random_move = RandomPlayer.uniform_random_genmove(g)
        black_groups, white_groups = update_unionfind(int_random_move, turn, g, black_groups, white_groups)
        status = winner(black_groups, white_groups)
        g.append(int_random_move)
        move_seq.append(intmove_to_raw(int_random_move))
        turn = (turn + 1) % 2
        if status == 0 or status == 1: return False

        # Strong play from U+1 till game ends
        self.strong_player.play_move_seq(move_seq)
        toplay="black" if turn==0 else "white"
        ans=self.strong_player.sendCommand("dfpn-solve-state "+toplay).strip()
        #print("move seq:", move_seq, "ans:",ans)
        if ans==toplay:
            return move_seq, 1.0
        else:
            return move_seq, -1.0

    def generate_one_example(self, param_time_limit1=None, param_time_limit2=None):
        self.fast_player.set_board_size(BOARD_SIZE)
        self.strong_player.set_board_size(BOARD_SIZE)
        if param_time_limit1:
            self.fast_player.sendCommand(param_time_limit1)

        if param_time_limit2:
            self.strong_player.sendCommand(param_time_limit2)

        U = np.random.randint(0, BOARD_SIZE ** 2-BOARD_SIZE)
        g = []
        move_seq=[]
        black_groups = unionfind()
        white_groups = unionfind()
        turn = 0

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
            move_seq.append(move)
            if (status == 0 or status == 1): return False

        # Random play at step U
        int_random_move = RandomPlayer.uniform_random_genmove(g)
        black_groups, white_groups = update_unionfind(int_random_move, turn, g, black_groups, white_groups)
        status = winner(black_groups, white_groups)
        g.append(int_random_move)
        move_seq.append(intmove_to_raw(int_random_move))

        example_state_player = turn
        turn = (turn + 1) % 2
        if status == 0 or status == 1: return False

        # Strong play from U+1 till game ends
        self.strong_player.play_move_seq(move_seq)
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

        return move_seq, R

def main(argv=None):
    exe1 = FLAGS.exe1_path+" 2>/dev/null"
    exe2 = FLAGS.exe2_path+" 2>/dev/null"
    print(exe1)
    eproducer=ExampleProducer(exe1, exe2, FLAGS.num)
    eproducer.produce_data(solving=True)

if __name__ == "__main__":
    tf.app.run()