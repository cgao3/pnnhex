from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from six.moves import xrange

import numpy as np
import tensorflow as tf
from read_data import INPUT_DEPTH, INPUT_WIDTH, BOARD_SIZE
from policygradient import PGNetwork
from supervised import SLNetwork

from unionfind import unionfind
from game_util import  *
from layer import Layer

from supervised import SL_MODEL_PATH

PG_MODEL_PATH="opponent_pool/model-10.ckpt"

VNET_BATCH_SIZE=64
NUM_EXAMPLES=100000
EXAMPLE_DIR="examples_valuenet/"

class SLNetPlayer(object):

    def __init__(self):
        self.data_node=tf.placeholder(dtype=tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        slnet=SLNetwork("sl_player")
        with tf.variable_scope("sl_player"):
            slnet.declare_layers()
        self.logits_node=slnet.model(self.data_node)

    def open_session(self):
        sess=tf.Session()
        self.sess=sess
        saver=tf.train.Saver()
        saver.restore(self.sess, SL_MODEL_PATH)

    def close_session(self):
        self.sess.close()

    def feedforward(self, input_tensor):
        logits = self.sess.run(self.logits_node, feed_dict={self.data_node: input_tensor})
        return logits

    def generate_move(self, input_tensor, intgamestate):
        logits=self.feedforward(input_tensor)
        return softmax_selection(logits, intgamestate)


class PGNetPlayer(object):
    def __int__(self):
        self.data_node = tf.placeholder(dtype=tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        pgnet= PGNetwork("pg_player")
        self.logits_node=pgnet.model(self.data_node)


    def open_session(self):
        sess=tf.Session()
        self.sess=sess
        saver=tf.train.Saver()
        saver.restore(sess, PG_MODEL_PATH)

    def close_session(self):
        self.sess.close()

    def feedforward(self, input_tensor):
        logits=self.sess.run(self.logits_node, feed_dict={self.data_node: input_tensor})
        return logits

    def generate_move(self, input_tensor, intgamestate):
        logits = self.feedforward(input_tensor)
        return softmax_selection(logits, intgamestate)


class RandomPlayer(object):

    def __init__(self):
        pass

    @staticmethod
    def uniform_random_genmove(intgamestate, boardsize):
        N=boardsize**2-1
        empty_positions=[i for i in range(N) if i not in intgamestate]
        return np.random.choice(empty_positions)

class ValueNet(object):

    def __init__(self, U=None):
        self.U=np.random.randint(0, BOARD_SIZE**2) if U==None else U

    def produce_data(self):
        self.slnet_player = SLNetPlayer()
        self.slnet_player.open_session()
        self.pgnet_player = PGNetPlayer()
        self.pgnet_player.open_session()
        self.input_tensor = np.zeros(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        count=0
        fout=open(EXAMPLE_DIR+"exam.dat","w+")
        while count < NUM_EXAMPLES:
            raw_game, label=self.generate_one_example()
            if (raw_game, label):
                count += 1
                for m in raw_game:
                    fout.write(m+" ")
                fout.write("\n"+label+"\n")
        fout.close()

        self.slnet_player.close_session()
        self.pgnet_player.close_session()

    def generate_one_example(self):
        U = np.random.randint(0, BOARD_SIZE ** 2)
        gamestate = []
        black_groups=unionfind()
        white_groups=unionfind()
        turn=0
        gamestatus=-1
        make_empty_board_tensor(self.input_tensor)

        #SL play to step U-1
        for i in range(0, U):
            intmove = self.slnet_player.generate_move(self.input_tensor, gamestate)
            update_tensor(self.input_tensor, turn, intmove)
            black_groups, white_groups=update_unionfind(intmove, turn, gamestate, black_groups, white_groups)
            gamestatus=winner(black_groups, white_groups)
            turn = (turn +1)%2
            gamestate.append(intmove)
            if (gamestatus == 0 or gamestate == 1): return

        #Random play at step U
        int_random_move=RandomPlayer.uniform_random_genmove(gamestate, BOARD_SIZE)
        update_tensor(self.input_tensor, turn, int_random_move)
        black_groups, white_groups=update_unionfind(int_random_move, turn, gamestate, black_groups, white_groups)
        gamestatus=winner(black_groups,white_groups)
        gamestate.append(int_random_move)
        example_state=list(gamestate)
        example_state_player=turn
        turn = (turn+1)%2
        if gamestatus == 0 or gamestatus == 1: return

        #PG play from U+1 till game ends
        while gamestatus==-1:
            intmove=self.pgnet_player.generate_move(self.input_tensor, gamestate)
            update_tensor(self.input_tensor, turn, intmove)
            black_groups,white_groups=update_unionfind(intmove, turn, gamestate, black_groups,white_groups)
            gamestatus=winner(black_groups,white_groups)
            turn = (turn + 1)%2
            gamestate.append(intmove)
        R = 1.0 if gamestatus == example_state_player else -1.0

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
        data_node=tf.placeholder(dtype=tf.float32, shape=(VNET_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
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
    vnet=ValueNet()

if __name__ == "__main__":
    tf.app.run()