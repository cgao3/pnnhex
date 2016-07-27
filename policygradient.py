
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from read_data import BOARD_SIZE, INPUT_WIDTH, INPUT_DEPTH, data_util
import numpy as np
from unionfind import unionfind
import tensorflow as tf
import time
import os
import random

from six.moves import xrange

from layer import *
from game_util import *

PG_GAME_BATCH_SIZE=128
PG_STATE_BATCH_SIZE=64

PGMODEL_NAME="pgmodel.ckpt"

from supervised import MODELS_DIR, SLMODEL_NAME, SLNetwork

MAX_NUM_MODEL_TO_KEEP=100

tf.flags.DEFINE_float("gamma", 0.95, "reward discount factor")
tf.flags.DEFINE_float("alpha", 0.02, "learning rate")

tf.flags.DEFINE_integer("max_iterations",1000, "max number of pg iterations")
tf.flags.DEFINE_integer("frequency",20, "after how many iterations updating the opponent model" )

FLAGS=tf.flags.FLAGS

class PGNetwork(object):

    def __init__(self):
        self.board_tensor = np.zeros(shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.float32)

    def select_model(self, models_dir):
        #list all models, randomly select one
        l2=[f for f in os.listdir(models_dir) if f.startswith(PGMODEL_NAME) and not f.endswith(".meta")]
        return os.path.join(models_dir, np.random.choice(l2))

    def play_one_batch_games(self, sess, otherSess, thisLogit, otherLogit, data_node, batch_game_size, batch_reward):

        this_win_count=0
        other_win_count=0

        this_player=random.randint(0,1)
        other_player=1-this_player
        games=[]
        for ind in range(batch_game_size):
            self.board_tensor.fill(0)
            make_empty_board_tensor(self.board_tensor)
            currentplayer = 0
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
            #print("steps ", count, "gamestatus ", gamestatus)
            R = 1.0/count if gamestatus == this_player else -1.0/count
            games.append([-1]+moves) #first hypothesisted action is -1
            batch_reward[ind]=R

        print("this player win: ", this_win_count, "other player win: ", other_win_count)
        return (games, this_win_count, other_win_count)

    def selfplay(self):
        data_node=tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        slnet=SLNetwork()
        slnet.declare_layers(num_hidden_layer=8)

        this_logits=slnet.model(data_node)
        saver=tf.train.Saver(max_to_keep=MAX_NUM_MODEL_TO_KEEP)
        print(this_logits.name)
        sl_model = os.path.join(MODELS_DIR, SLMODEL_NAME)
        sess = tf.Session()
        saver.restore(sess, sl_model)

        with tf.variable_scope("other_nn_player"):
            slnet2=SLNetwork()
            slnet2.declare_layers(num_hidden_layer=8)
            other_logits=slnet2.model(data_node)
            #use non-scoped name to restore those variables
            var_dict2 = {slnet.input_layer.weight.op.name: slnet2.input_layer.weight,
                    slnet.input_layer.bias.op.name: slnet2.input_layer.bias}
            for i in xrange(slnet2.num_hidden_layer):
                var_dict2[slnet.conv_layer[i].weight.op.name] = slnet2.conv_layer[i].weight
                var_dict2[slnet.conv_layer[i].bias.op.name] = slnet2.conv_layer[i].bias
            saver2 = tf.train.Saver(var_list=var_dict2)

            otherSess = tf.Session()
            saver2.restore(otherSess, sl_model)

        batch_game_size=128

        batch_reward_node = tf.placeholder(dtype=np.float32, shape=(PG_STATE_BATCH_SIZE,))
        batch_data_node = tf.placeholder(dtype=np.float32, shape=(PG_STATE_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        batch_label_node = tf.placeholder(shape=(PG_STATE_BATCH_SIZE,), dtype=np.int32)

        batch_rewards = np.ndarray(dtype=np.float32, shape=(PG_STATE_BATCH_SIZE,))
        batch_data = np.ndarray(dtype=np.float32, shape=(PG_STATE_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        batch_labels = np.ndarray(shape=(PG_STATE_BATCH_SIZE,), dtype=np.int32)

        game_rewards=np.ndarray(shape=(batch_game_size,), dtype=np.float32)

        tf.get_variable_scope().reuse_variables()
        logit = slnet.model(batch_data_node)
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, batch_label_node)
        loss = tf.reduce_mean(tf.mul(batch_reward_node, entropy))
        opt = tf.train.GradientDescentOptimizer(FLAGS.alpha / batch_game_size)
        opt_op=opt.minimize(loss)

        win1=0
        win2=0
        ite=0
        g_step=0

        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            print("no SL model? creating pg model dir")
        else:
            for f in os.listdir(MODELS_DIR):
                if f.startswith(PGMODEL_NAME):
                    try:
                        os.remove(os.path.join(MODELS_DIR,f))
                    except OSError as e:
                        print(e.strerror)
            print("removing old models in pg dir")

        while ite < FLAGS.max_iterations:
            start_batch_time = time.time()
            games,_tmp1,_tmp2=self.play_one_batch_games(sess,otherSess, this_logits,other_logits,data_node,batch_game_size, game_rewards)
            win1 += _tmp1
            win2 += _tmp2
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
                            batch_rewards[k]=R*sign*(FLAGS.gamma**(len(games[i])-2-j))
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

            print("time cost for one batch of %d games"%PG_GAME_BATCH_SIZE, time.time()-start_batch_time)
            ite += 1
            if ite % FLAGS.frequency==0:
                saver.save(sess, os.path.join(MODELS_DIR, PGMODEL_NAME), global_step=g_step)
                pg_model=self.select_model(MODELS_DIR)
                saver2.restore(otherSess, pg_model)
                g_step +=1
                print("Replce opponenet with new model, ", g_step)
                print(pg_model)
            ite += 1
        print("In total, this win", win1, "opponenet win", win2)
        otherSess.close()
        sess.close()

    def test_play(self, batch_game_size=128):
        data = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        otherSess = tf.Session()
        otherLogit = self.model(data)
        saver = tf.train.Saver()
        saver.restore(otherSess, os.path.join(MODELS_DIR, SLMODEL_NAME))

        tf.get_variable_scope().reuse_variables()
        sess = tf.Session()
        thisLogit = self.model(data)
        saver.restore(sess, os.path.join(MODELS_DIR, SLMODEL_NAME))

        game_rewards = np.ndarray(shape=(batch_game_size,), dtype=np.float32)
        this_win=0
        other_win=0
        for _ in range(3):
            _, this, that=self.play_one_batch_games(sess, otherSess, thisLogit, otherLogit, data, batch_game_size, game_rewards)
            this_win  += this
            other_win +=that

        print("This player wins ", this_win, "that player wins", other_win)

def main(argv=None):
    pg=PGNetwork()
    pg.selfplay()

if __name__ == "__main__":
    tf.app.run()
