
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from read_data import BOARD_SIZE, INPUT_WIDTH, INPUT_DEPTH, data_util
import numpy as np
from unionfind import unionfind
import tensorflow as tf
import time
import os

from six.moves import xrange

from layer import *
from game_util import *

PG_GAME_BATCH_SIZE=128
PG_STATE_BATCH_SIZE=64

tf.flags.DEFINE_string("pg_model_dir", "savedPGModel/", "where the model is saved")
tf.flags.DEFINE_string("supervised_model_path", "savedModel/model.ckpt", "where the supervised learning model is")

tf.flags.DEFINE_float("gamma", 0.95, "reward discount factor")
tf.flags.DEFINE_float("alpha", 0.1, "learning rate")

tf.flags.DEFINE_integer("max_num_to_keep", 100, "max number of models kept in pg_model_dir")

tf.flags.DEFINE_integer("max_pg_iteration", 10, "max number of pg iterations to train")
#num_pg_batch_per_ite*batch_game_size is the number training examples per pg_iteration
tf.flags.DEFINE_integer("num_batch_game_per_ite", 5, "how many batch games per iteration" )

FLAGS=tf.flags.FLAGS

class PGNetwork(object):

    def __init__(self, name):
        self.bashline=0.0
        self.board_tensor = np.zeros(shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.float32)

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

    def select_model(self, models_dir):
        #list all models, randomly select one
        l2=[f for f in os.listdir(models_dir) if f.startswith("model.ckpt") and not f.endswith(".meta")]
        return os.path.join(models_dir, np.random.choice(l2))

    def play_one_batch_games(self, sess, otherSess, thisLogit, otherLogit, data_node, batch_game_size, batch_reward):
        this_player = 0
        other_player = 1
        this_win_count=0
        other_win_count=0
        games=[]
        for ind in range(batch_game_size):
            self.board_tensor.fill(0)
            make_empty_board_tensor(self.board_tensor)
            currentplayer = this_player
            gamestatus = -1
            black_group = unionfind()
            white_group = unionfind()
            count = 0
            moves=[]
            while (gamestatus == -1):
                if (currentplayer == this_player):
                    logit = sess.run(thisLogit, feed_dict={data_node: self.board_tensor})
                else:
                    logit = otherSess.run(otherLogit, feed_dict={data_node: self.board_tensor})
                action = softmax_selection(logit, moves)
                update_tensor(self.board_tensor, currentplayer, action)
                black_group, white_group = update_unionfind(action, currentplayer, moves, black_group, white_group)
                currentplayer = other_player if currentplayer == this_player else this_player
                gamestatus = winner(black_group, white_group)
                moves.append(action)
                count += 1
                #print(count, "action ", action)
            if(gamestatus==this_player): this_win_count += 1
            else: other_win_count += 1
            print("steps ", count, "gamestatus ", gamestatus)
            R = 1.0 if gamestatus == this_player else -1.0
            games.append([-1]+moves) #first hypothesisted action is -1
            batch_reward[ind]=R*count

        print("this player win: ", this_win_count, "other player win: ", other_win_count)
        return (games, this_win_count, other_win_count)

    def selfplay(self, max_iteration):
        data=tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        otherNeuroPlayer = PGNetwork("other_player")
        otherSess = tf.Session()
        otherLogit=otherNeuroPlayer.model(data)
        saver=tf.train.Saver(max_to_keep=FLAGS.max_num_to_keep)
        sl_model=FLAGS.supervised_model_path
        saver.restore(otherSess, sl_model)

        sess=tf.Session()
        thisLogit=self.model(data)
        saver.restore(sess, sl_model)
        batch_game_size=128
        opt = tf.train.GradientDescentOptimizer(FLAGS.alpha/batch_game_size)
        #opt =tf.train.AdamOptimizer()

        batch_reward_node = tf.placeholder(dtype=np.float32, shape=(PG_STATE_BATCH_SIZE,))
        batch_data_node = tf.placeholder(dtype=np.float32, shape=(PG_STATE_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        batch_label_node = tf.placeholder(shape=(PG_STATE_BATCH_SIZE,), dtype=np.int32)

        batch_rewards = np.ndarray(dtype=np.float32, shape=(PG_STATE_BATCH_SIZE,))
        batch_data = np.ndarray(dtype=np.float32,
                                         shape=(PG_STATE_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        batch_labels = np.ndarray(shape=(PG_STATE_BATCH_SIZE,), dtype=np.int32)

        game_rewards=np.ndarray(shape=(batch_game_size,), dtype=np.float32)

        logit = self.model(batch_data_node)
        loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, batch_label_node)
        loss = tf.reduce_mean(tf.mul(batch_reward_node, loss1))
        opt_op=opt.minimize(loss)
        win1=0
        win2=0

        num_batch=FLAGS.num_batch_game_per_ite
        batch_count=0
        g_step=0

        while batch_count < num_batch:
            games,_tmp1,_tmp2=self.play_one_batch_games(sess,otherSess, thisLogit,otherLogit,data,batch_game_size, game_rewards)
            win1 += _tmp1
            win2 += _tmp2
            start_batch_time=time.time()
            data_tool=data_util(games, PG_STATE_BATCH_SIZE, batch_data, batch_labels)
            data_tool.disable_symmetry_checking()
            offset1=0; offset2=0; nepoch=0
            while nepoch <1 :
                o1,o2,next_epoch=data_tool.prepare_batch(offset1,offset2)
                if(next_epoch):
                    nepoch += 1
                k=0
                batch_rewards.fill(0)
                new_o1, new_o2=0,0
                while k < PG_STATE_BATCH_SIZE:
                    for i in range(offset1, batch_game_size):
                        R=game_rewards[i]
                        sign=1 if offset2 % 2 ==0 else -1
                        for j in range(offset2, len(games[i])-1):
                            batch_rewards[k]=1.0/R*sign*(FLAGS.gamma**(len(games[i])-2-j))
                            sign=-sign
                            k += 1
                            if(k>=PG_STATE_BATCH_SIZE):
                                new_o1=i
                                new_o2=j+1
                                break
                        offset2=0
                        if(k>=PG_STATE_BATCH_SIZE):
                            break
                    if k<PG_STATE_BATCH_SIZE:
                        offset1, offset2 = 0,0
                assert(new_o1==o1 and new_o2==o2)
                offset1, offset2=o1,o2
                sess.run(opt_op, feed_dict={batch_data_node:batch_data, batch_label_node:batch_labels, batch_reward_node: batch_rewards})

            print("time cost for all batch of games", time.time()-start_batch_time)

            batch_count += 1
            if batch_count == num_batch:

                saver.save(sess, os.path.join(FLAGS.pg_model_dir, "model.ckpt"), global_step=g_step)
                pg_model=self.select_model(FLAGS.pg_model_dir)
                saver.restore(otherSess, pg_model)
                g_step +=1
                if g_step < max_iteration:
                    batch_count=0
                print("Replce opponenet with new model, ", g_step)
                print(pg_model)

        print("In total, this win", win1, "opponenet win", win2)
        otherSess.close()
        sess.close()

    def test_play(self):
        data = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        otherNeuroPlayer = PGNetwork("other_player")
        otherSess = tf.Session()
        otherLogit = otherNeuroPlayer.model(data)
        saver = tf.train.Saver()
        saver.restore(otherSess, "savedModel/model.ckpt")

        sess = tf.Session()
        thisLogit = self.model(data)
        saver.restore(sess, "savedModel/model.ckpt")
        batch_game_size = 128

        game_rewards = np.ndarray(shape=(batch_game_size,), dtype=np.float32)
        this_win=0
        other_win=0
        for _ in range(3):
            _, this, that=self.play_one_batch_games(sess, otherSess, thisLogit, otherLogit, data, batch_game_size, game_rewards)
            this_win += this
            other_win +=that

        print("This player wins ", this_win, "that player wins", other_win)


def main(argv=None):
    pg=PGNetwork("pg training")
    max_ite=FLAGS.max_pg_iteration
    pg.selfplay(max_ite)

if __name__ == "__main__":
    tf.app.run()
