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

VALUE_NET_EXAMPLES_DIR="examples/"
VALUE_NET_BATCH_SIZE=64

NUM_EXAMPLES=1e6

class RandomPlayer(object):

    def __init__(self):
        pass

    @staticmethod
    def uniform_random_genmove(intgamestate):
        N=BOARD_SIZE**2
        empty_positions=[i for i in range(N) if i not in intgamestate]
        return np.random.choice(empty_positions)

class ValueNet(object):

    def __init__(self, exe1, exe2, max_time1=0.5, max_time=1):
        self.fast_player=WrapperAgent(exe1)
        self.strong_player=WrapperAgent(exe2)

    def produce_data(self, num_examples=10000):
        count=0
        fout=open(VALUE_NET_EXAMPLES_DIR+"exam.dat","w+")
        while count < num_examples:
            raw_game, label=self.generate_one_example()
            if (raw_game, label):
                count += 1
                for m in raw_game:
                    fout.write(m+" ")
                fout.write("\n"+label+"\n")
        fout.close()

    def generate_one_example(self):
        self.fast_player.clear_board()
        self.fast_player.sendCommand("param_wolve max_time 0.5")

        self.strong_player.clear_board()
        self.strong_player.sendCommand("param_wolve max_time 1")
        U = np.random.randint(0, BOARD_SIZE ** 2)
        g = []
        black_groups=unionfind()
        white_groups=unionfind()
        turn=0
        status=-1
        #Fast Player play to U
        for i in range(0, U):
            move = self.fast_player.genmove_black() if turn ==0 else self.fast_player.genmove_white()
            intmove=raw_move_to_int(move)
            black_groups, white_groups=update_unionfind(intmove, turn, g, black_groups, white_groups)
            status=winner(black_groups, white_groups)
            turn = (turn +1)%2
            g.append(intmove)
            if (status == 0 or status == 1): return

        #Random play at step U
        int_random_move=RandomPlayer.uniform_random_genmove(g)
        black_groups, white_groups=update_unionfind(int_random_move, turn, g, black_groups, white_groups)
        status=winner(black_groups,white_groups)
        g.append(int_random_move)
        example_state=list(g)
        example_state_player=turn
        turn = (turn+1)%2
        if status == 0 or status == 1: return

        #Strong play from U+1 till game ends
        while status==-1:
            move=self.strong_player.genmove_black() if turn ==0 else self.strong_player.genmove_white()
            intmove=raw_move_to_int(move)
            black_groups,white_groups=update_unionfind(intmove, turn, g, black_groups,white_groups)
            status=winner(black_groups,white_groups)
            turn = (turn + 1)%2
            g.append(intmove)
        R = 1.0 if status == example_state_player else -1.0

        return self.build_example(example_state, R)

    #label is win or loss, +1 or -1, OR modified game reward
    def build_example(self, intgamestate, label):
        raw_game=[]
        for m in intgamestate:
            rawmove=intmove_to_raw(m)
            raw_game.append(rawmove)
        return (raw_game, label)

    def regression(self):
        
        pass

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