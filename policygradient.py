
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from read_data import BOARD_SIZE, INPUT_WIDTH, INPUT_DEPTH
import numpy as np
from unionfind import unionfind
import tensorflow as tf
import time

from layer import *
from game_util import *

PG_BATCH_SIZE=128
NORTH_EDGE=-1
SOUTH_EDGE=-3

EAST_EDGE=-2
WEST_EDGE=-4


class PGNetwork(object):

    def __init__(self, name):
        self.bashline=0.0
        with tf.variable_scope(name):
            self.declare_layers(num_hidden_layer=8)

    #the same structure as supervised network
    def declare_layers(self, num_hidden_layer):
        self.num_hidden_layer = num_hidden_layer
        self.input_layer = Layer("input_layer", paddingMethod="VALID")
        self.conv_layer = [None] * num_hidden_layer

        for i in range(num_hidden_layer):
            self.conv_layer[i] = Layer("conv%d_layer" % i)

    def model(self, data_node, kernal_size=(3, 3), kernal_depth=80):
        output = [None] * self.num_hidden_layer
        weightShape0 = kernal_size + (INPUT_DEPTH, kernal_depth)
        output[0] = self.input_layer.convolve(data_node, weight_shape=weightShape0, bias_shape=(kernal_depth,))

        weightShape = kernal_size + (kernal_depth, kernal_depth)
        for i in range(self.num_hidden_layer - 1):
            output[i + 1] = self.conv_layer[i].convolve(output[i], weight_shape=weightShape, bias_shape=(kernal_depth,))

        logits = self.conv_layer[self.num_hidden_layer - 1].one_filter_out(output[self.num_hidden_layer - 1], BOARD_SIZE)
        tf.get_variable_scope().reuse_variables()
        #self._init=tf.initialize_all_variables()
        return logits

    #input is raw score such as [-20,30,10]
    def softmax_selection(self, logits, currentstate):
        empty_positions=[i for i in range(BOARD_SIZE**2) if i not in currentstate]
        logits=np.squeeze(logits)
        #print("empty positions:", empty_positions)
        #print(logits)
        effective_logits=[logits[i] for i in empty_positions]
        max_value=np.max(effective_logits)
        effective_logits=effective_logits - max_value
        effective_logits=np.exp(effective_logits)
        sum_value=np.sum(effective_logits)
        prob=effective_logits/sum_value
        v=np.random.random()
        sum_v=0.0
        action=None
        for i, e in enumerate(prob):
            sum_v += e
            if(sum_v >= v):
                action=i
                break
        return empty_positions[action]

    def update_tensor(self, tensor, player, intmove):
        x,y=intmove//BOARD_SIZE+1, intmove%BOARD_SIZE+1
        tensor[0, x, y, player] = 1
        tensor[0, x, y, 2] = 0
        return tensor

    def init_tensor(self):
        t = np.zeros(shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.int16)
        t[0, 0:INPUT_WIDTH, 0, 0] = 1
        t[0, 0:INPUT_WIDTH, INPUT_WIDTH - 1, 0] = 1

        t[0, 0, 1:INPUT_WIDTH - 1, 1] = 1
        t[0, INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 1] = 1

        t[0, 1:INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 2] = 1
        return t

    #build batch from a single game
    def build_batch_from_game(self, batch_states, batch_label, g):
        from read_data import data_util
        data=data_util()
        data.batch_states=batch_states
        data.batch_labels=batch_label
        data.batchsize=len(batch_label)
        data.games=[g]
        o1,o2, next_batch=data.prepare_batch(0,0)
        print("offset ",o1,o2,"next batch", next_batch)
    def select_model(self, models_location_dir):
        #list all models, randomly select one

        return "savedModel/model.ckpt"
    def play_one_batch_games(self, sess, otherSess, thisLogit, otherLogit, data_node, batch_game_size):
        this_player = 0
        other_player = 1
        grad_vals_list = []
        grad_placeholder_list = []
        apply_grad_placeholder_list = []
        this_win_count=0
        other_win_count=0
        Rewards=[]
        games=[]
        for one_game in range(batch_game_size):
            moves = []
            input_tensor = self.init_tensor()
            currentplayer = this_player
            gamestatus = -1
            black_group = unionfind()
            white_group = unionfind()
            count = 0
            while (gamestatus == -1):
                if (currentplayer == this_player):
                    logit = sess.run(thisLogit, feed_dict={data_node: input_tensor.astype(np.float32)})
                else:
                    logit = otherSess.run(otherLogit, feed_dict={data_node: input_tensor.astype(np.float32)})
                action = self.softmax_selection(logit, moves)
                moves.append(action)
                input_tensor = self.update_tensor(input_tensor, currentplayer, action)
                black_group, white_group = \
                    self.update_unionfind(action, currentplayer, moves, black_group, white_group)
                currentplayer = other_player if currentplayer == this_player else this_player
                gamestatus = self.winner(black_group, white_group)
                count += 1
                #print(count, "action ", action)
            if(gamestatus==this_player): this_win_count += 1
            else: other_win_count += 1
            R = 1.0 if gamestatus == this_player else -1.0
            Rewards.append(R)
            games.append([-1]+moves)
        print("this player win: ", this_win_count, "other player win: ", other_win_count)
        return (games, Rewards)

    def selfplay(self):
        data=tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        label_node=tf.placeholder(dtype=tf.int32, shape=(1,))
        otherNeuroPlayer = PGNetwork("other_player")
        otherSess = tf.Session()
        otherLogit=otherNeuroPlayer.model(data)
        saver=tf.train.Saver()
        saver.restore(otherSess, "savedModel/model.ckpt")

        sess=tf.Session()
        thisLogit=self.model(data)
        saver.restore(sess,"savedModel/model.ckpt")
        batch_game_size=5
        opt = tf.train.GradientDescentOptimizer(0.001/batch_game_size)
        #sess.run(tf.initialize_all_variables())
        logit=self.model(data)
        R_node=tf.placeholder(dtype=tf.float32)
        loss=tf.mul(R_node, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label_node)))
        for _ in range(30):
            games, rewards=self.play_one_batch_games(sess,otherSess, thisLogit,otherLogit,data,batch_game_size)
            for i in range(batch_game_size):
                R=rewards[i]
                game=games[i]
                game_length=len(game)-1
                batch_data = np.ndarray(dtype=np.float32, shape=(game_length, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
                batch_label=np.ndarray(dtype=np.int32, shape=(game_length,))
                self.build_batch_from_game(batch_data, batch_label, game)
                sign=-1
                for j in range(game_length):
                    sess.run(opt.minimize(loss), feed_dict={R_node:R*sign, data: batch_data[j:j+1],label_node:batch_label[j:j+1]})
                    sign=sign*(-1)

        otherSess.close()
        sess.close()

    def play_test(self, model_path="savedModel/model.ckpt"):
        start_time=time.time()
        sess=tf.Session()
        tensor=self.init_tensor()
        logi=self.model(tensor.astype(np.float32))
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("model loaded")
        black_group=unionfind()
        white_group=unionfind()
        gamestatus=-1
        g=[]
        currentplayer=0
        count=0

        print("game begins")
        print(state_to_str(g) )
        while gamestatus==-1:
            lg=sess.run(logi)
            A=self.softmax_selection(lg, g)
            print("Action: ", count, "is ", (A//BOARD_SIZE, A%BOARD_SIZE))
            g.append(A)
            black_group,white_group=self.update_unionfind(A,currentplayer,g,black_group,white_group)
            gamestatus = self.winner(black_group, white_group)
            logi = self.model(tensor.astype(np.float32))
            tensor = self.update_tensor(tensor, currentplayer, A)
            currentplayer = next_player(currentplayer)
            count +=1
        sess.close()
        print("game ends winstatus ", gamestatus)
        print(state_to_str(g))
        print("print ends time cost: ", time.time()-start_time)

    def winner(self, black_group, white_group):
        if(black_group.connected(NORTH_EDGE,SOUTH_EDGE)):
            return 0
        elif(white_group.connected(WEST_EDGE,EAST_EDGE)):
            return 1
        else: return -1

    def update_unionfind(self, intmove, player, board, black_group, white_group):
        x,y=intmove//BOARD_SIZE, intmove%BOARD_SIZE
        neighbors=[]
        pattern=[(-1,0),(0,-1),(0,1),(1,0),(-1,1),(1,-1)]
        for p in pattern:
            x1,y1=p[0]+x,p[1]+y
            if 0 <= x1 < BOARD_SIZE and 0 <= y1 < BOARD_SIZE:
                neighbors.append((x1,y1))
        if(player==0):

            if(y==0):
                black_group.join(intmove,NORTH_EDGE)
            if(y==BOARD_SIZE-1):
                black_group.join(intmove,SOUTH_EDGE)

            for m in neighbors:
                m2=m[0]*BOARD_SIZE+m[1]
                if(m2 in board and board.index(m2) % 2 == player):
                    black_group.join(m2, intmove)
        else:

            if(x==0):
                white_group.join(intmove, WEST_EDGE)
            if(x==BOARD_SIZE-1):
                white_group.join(intmove, EAST_EDGE)

            for m in neighbors:
                im = m[0] * BOARD_SIZE + m[1]
                if (im in board and board.index(im) % 2 == player):
                    white_group.join(im, intmove)
        #print(black_group.parent)
        return (black_group, white_group)

if __name__ == "__main__":
    pgtest = PGNetwork("hello")
    pgtest.selfplay()
