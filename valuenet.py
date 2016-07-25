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
from game_util import *
import os
from supervised import SLNetwork, MODELS_DIR, SLMODEL_NAME


VALUE_NET_BATCH_SIZE=64

EVAL_SIZE=100
VALUE_NET_MODEL_PATH="valuemodel/valuenet_model.ckpt"

tf.flags.DEFINE_string("train_example_path", "vexamples/"+repr(BOARD_SIZE)+"x"+repr(BOARD_SIZE)+".dat", "train examples path")
tf.flags.DEFINE_string("eval_example_path", "vexamples/"+"test_"+repr(BOARD_SIZE+"x"+repr(BOARD_SIZE+".dat")))

class ValueNet(object):

    def __init__(self):
        self.slnet=SLNetwork()
        self.slnet.declare_layers(num_hidden_layer=8)
        self.batch_states=np.ndarray(dtype=np.float32, shape=(VALUE_NET_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.batch_label=np.ndarray(dtype=np.float32, shape=(VALUE_NET_BATCH_SIZE,))

        self.eval_batch_states = np.ndarray(dtype=np.float32, shape=(EVAL_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.eval_batch_label = np.ndarray(dtype=np.float32, shape=(EVAL_SIZE,))

    def train(self, train_states_data_path, eval_states_data_path):
        train_raw_states=[]
        with open(train_states_data_path, "r") as f:
            for line in f:
                train_raw_states.append(line.split())

        eval_raw_states=[]
        with open(eval_states_data_path, "r") as f:
            for line in f:
                eval_raw_states.append(line.split())
        print("train and eval data loaded..")

        num_epochs=10
        epoch_count=0
        offset=0

        batch_states_node=tf.placeholder(dtype=tf.float32, shape=(VALUE_NET_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        batch_targets_node=tf.placeholder(dtype=tf.float32, shape=(VALUE_NET_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))

        eval_states_node = np.ndarray(dtype=np.float32,shape=(EVAL_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        eval_targets_node = np.ndarray(dtype=np.float32, shape=(EVAL_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))

        output=self.model(batch_states_node)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(output, batch_targets_node))))
        opt_op=tf.train.RMSPropOptimizer().minimize(rmse)

        tf.get_variable_scope().reuse_variables()
        eval_output = self.model(eval_states_node)
        eval_rmse=tf.sqrt(tf.reduce_mean(tf.square(tf.sub(eval_output, eval_targets_node))))
        eval_opt_op=tf.train.RMSPropOptimizer().minimize(eval_rmse)

        sess=tf.Session()
        sess.run(tf.initialize_all_variables())

        var_dict = {self.slnet.input_layer.weight.op.name: self.slnet.input_layer.weight,
                     self.slnet.input_layer.bias.op.name: self.slnet.input_layer.bias}

        for i in xrange(self.slnet.num_hidden_layer):
            var_dict[self.slnet.conv_layer[i].weight.op.name] = self.slnet.conv_layer[i].weight
            var_dict[self.slnet.conv_layer[i].bias.op.name] = self.slnet.conv_layer[i].bias
        saver = tf.train.Saver(var_list=var_dict)
        sl_model = os.path.join(MODELS_DIR, SLMODEL_NAME)
        #restore variables from SL model
        saver.restore(sess, sl_model)

        while epoch_count < num_epochs:
            offset, next_epoch=self.prepare_batch(train_raw_states, offset, self.batch_states, self.batch_label,
                                                  batchsize=VALUE_NET_BATCH_SIZE)
            rmse_=sess.run(opt_op, feed_dict={batch_states_node:self.batch_states,
                                              batch_targets_node:self.batch_label})
            print("epoch", epoch_count, "RMSE ", rmse_)

            if (epoch_count+1) % 10 == 0:
                self.prepare_batch(eval_raw_states,0,self.eval_batch_states, self.eval_batch_label, EVAL_SIZE)
                eval_rmse_ = sess.run(eval_opt_op, feed_dict=
                {eval_states_node: self.eval_batch_states, eval_targets_node:self.eval_batch_label})
                print("eval RMSE:", eval_rmse_)

            if next_epoch:
                epoch_count += 1
        if not os.path.exists(VALUE_NET_MODEL_PATH):
            print("creating valuenet model directory")
            os.mkdir(os.path.dirname(VALUE_NET_MODEL_PATH))
        
        saver.save(VALUE_NET_MODEL_PATH)
        sess.close()

    #the ith state, build kth batch tensor label
    def build_example(self, raw_states, ith, batch_tensor, batch_label, kth):
        batch_label[kth]=raw_states[ith][-1]
        make_kth_empty_tensor_in_batch(batch_tensor=batch_tensor, kth=kth)
        turn=0
        for j in range(len(raw_states[ith])-1):
            move=raw_states[ith][j]
            intmove=raw_move_to_int(move)
            update_kth_tensor_in_batch(batch_tensor=batch_tensor,kth=kth,player=turn,intmove=intmove)
            turn = (turn+1)%2

    def prepare_batch(self, raw_states, offset, batch_tensor, batch_label, batchsize):
        count=0
        new_offset=offset
        next_epoch=False
        while count<batchsize:
            for i in xrange(offset, len(raw_states)):
                self.build_example(raw_states, i,batch_tensor, batch_label, count)
                count += 1
                if count >= batchsize:
                    new_offset=i+1
                    break
            if count < batchsize:
                offset=0
                next_epoch=True
        return new_offset, next_epoch

    def model(self, data_node):
        logits = self.slnet.model(data_node)
        last_layer=Layer("value_net_output_layer", paddingMethod="SAME", reuse_var=False)
        value=last_layer.value_estimation(logits, 80)
        return value

def main(argv=None):
    vnet=ValueNet()
    train_path=tf.flags.FLAGS.train_example_path
    test_path=tf.flags.FLAGS.test_example_path
    vnet.train(train_path, test_path)

if __name__ == "__main__":
    tf.app.run()