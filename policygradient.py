
from read_data import BOARD_SIZE, INPUT_WIDTH, INPUT_DEPTH

import numpy as np
from unionfind import unionfind
import tensorflow as tf
from supervised import SLNetwork
from layer import *

PG_BATCH_SIZE=128
batch_games=[]

NORTH_EDGE=-1
SOUTH_EDGE=-3

EAST_EDGE=-2
WEST_EDGE=-4

class PGNetwork(object):

    def __init__(self, name):
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
        print(empty_positions)
        print(logits)
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

    def forwardpass(self, input_tensor, sess):
        data = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        logits=self.model(data)
        sess.run(tf.initialize_all_variables())
        return sess.run(logits, feed_dict={data:input_tensor})

    def game_status(self):

        pass

    def update_tensor(self):

        return

    def init_tensor(self):
        t = np.zeros(shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.int16)
        t[0, 0:INPUT_WIDTH, 0, 0] = 1
        t[0, 0:INPUT_WIDTH, INPUT_WIDTH - 1, 0] = 1

        t[0, 0, 1:INPUT_WIDTH - 1, 1] = 1
        t[0, INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 1] = 1

        t[0, 1:INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 2] = 1
        return t

    def selfplay(self, other_player):

        state=[]
        input_tensor=0
        player=PGNetwork("other_player")
        sessOther=tf.Session()
        sess=tf.Session()
        for batch_t in range(1):
            state=[]
            input_tensor=self.init_tensor()
            currentplayer=0
            gamestatus=-1
            black = unionfind()
            white = unionfind()
            tensors=[]
            while (gamestatus==-1):
                tensors.append(input_tensor)
                if(currentplayer==0):
                    logit = self.forwardpass(input_tensor, sess)
                else:
                    logit = player.forwardpass(input_tensor, sessOther)
                action=self.softmax_selection(logit,state)
                state.append(action)
                x=action//BOARD_SIZE+1
                y=action%BOARD_SIZE+1
                input_tensor[1,x,y,currentplayer]=1
                input_tensor[1,x,y,2]=0
                self.update_unionfind((x-1,y-1), currentplayer, state, black, white)
                currentplayer = currentplayer + 1
                currentplayer %= 2
                gamestatus=self.winner(black,white)
            R=1.0 if gamestatus==0 else -1.0
            batch_label=np.array(state, dtype=np.int32)
            batch_tensors=np.array(tensors, dtype=np.float32)
            logit=self.model(batch_tensors)
            loss=tf.mul(tf.Variable(R), tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logit, batch_label)))
            opt=tf.train.AdamOptimizer()
            opt_op=opt.minimize(loss)
            sess.run(opt_op)

        sessOther.close()
        sess.close()

    def play_test(self, model_path="savedModel/model.ckpt"):
        sess=tf.Session()
        tensor=self.init_tensor()
        logit=self.model(tensor)
        sess.run(tf.initialize_all_variables())
        lg=sess.run(logit)
        print(lg)
        sess.close()
    def winner(self, black_group, white_group):
        if(black_group.connected(NORTH_EDGE,SOUTH_EDGE)):
            return 0
        elif(white_group.connected(WEST_EDGE,EAST_EDGE)):
            return 1
        else: return -1

    def update_unionfind(self, move, player, board, black_group, white_group):
        x,y=move
        neighbors=[]
        pattern=[(-1,0),(1,0),(0,1),(1,0),(-1,1),(1,-1)]
        for p in pattern:
            x1,y1=p[0]+x,p[1]+y
            if(x1 >=0 and x1 <BOARD_SIZE and y1 >=0 and y1 < BOARD_SIZE):
                neighbors.append((x1,y1))

        if(player==0):
            if(y==0):
                black_group.join((x,y),NORTH_EDGE)
            if(y==BOARD_SIZE-1):
                black_group.join((x,y),SOUTH_EDGE)

            for m in neighbors:
                intmove=m[0]*BOARD_SIZE+m[1]
                if(intmove in board and board.index(intmove)%2==player):
                    black_group.join(m, (x,y))
        else:
            if(x==0):
                white_group.join((x,y), WEST_EDGE)
            if(x==BOARD_SIZE-1):
                white_group.join((x,y), EAST_EDGE)

            for m in neighbors:
                intmove = m[0] * BOARD_SIZE + m[1]
                if (intmove in board and board.index(intmove) % 2 == player):
                    white_group.join(m, (x, y))
        return (black_group, white_group)


if __name__ == "__main__":
    pgtest = PGNetwork("hello")
    logit=np.random.rand(1,5)

    moves=[1,3]
    a=pgtest.softmax_selection(logit, moves)
    print("action selected:", a)