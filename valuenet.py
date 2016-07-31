from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from six.moves import xrange

import numpy as np
from game_util import *
import tensorflow as tf

from layer import Layer
from game_util import *
import os
from supervised import MODELS_DIR, SLMODEL_NAME


VALUE_NET_BATCH_SIZE=32

EVAL_SIZE=450
VALUE_NET_MODEL_PATH="valuemodel/valuenet_model.ckpt"
VALUE_NET_MODEL_DIR=os.path.dirname(VALUE_NET_MODEL_PATH)

tf.flags.DEFINE_string("train_example_path", "vexamples/"+repr(BOARD_SIZE)+"x"+repr(BOARD_SIZE)+".txt", "train examples path")
tf.flags.DEFINE_string("test_example_path", "vexamples/"+"test"+repr(BOARD_SIZE)+"x"+repr(BOARD_SIZE)+".txt", "test examples path")
tf.flags.DEFINE_string("summaries_dir","/tmp/valuenet_logs", "where the summaries are")
tf.flags.DEFINE_integer("n_epoches", 500, "number of epoches to train valuenet")
tf.flags.DEFINE_float("alpha", 0.001, "initial learning rate")
FLAGS=tf.flags.FLAGS

class ValueNet(object):

    def __init__(self):
        self.declare_layers(num_hidden_layer=6)


    def train(self, train_states_data_path, eval_states_data_path, num_epoch=100):
        self.batch_states = np.ndarray(dtype=np.float32, shape=(VALUE_NET_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.batch_label = np.ndarray(dtype=np.float32, shape=(VALUE_NET_BATCH_SIZE,))

        self.eval_batch_states = np.ndarray(dtype=np.float32, shape=(EVAL_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        self.eval_batch_label = np.ndarray(dtype=np.float32, shape=(EVAL_SIZE,))

        train_raw_states=[]
        with open(train_states_data_path, "r") as f:
            for line in f:
                train_raw_states.append(line.split())

        eval_raw_states=[]
        with open(eval_states_data_path, "r") as f:
            for line in f:
                eval_raw_states.append(line.split())
        print("train and eval data loaded..")
        print(len(eval_raw_states))
        epoch_count=0
        offset=0

        batch_states_node=tf.placeholder(dtype=tf.float32, shape=(VALUE_NET_BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        batch_targets_node=tf.placeholder(dtype=tf.float32, shape=(VALUE_NET_BATCH_SIZE,))

        eval_states_node = tf.placeholder(dtype=np.float32, shape=(EVAL_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        eval_targets_node = tf.placeholder(dtype=np.float32, shape=(EVAL_SIZE,))
        keep_prob_node=tf.placeholder(tf.float32)

        output=self.model(batch_states_node, keep_prob_node=keep_prob_node)
        mse_train = tf.reduce_mean(tf.square(tf.sub(output, batch_targets_node)))
        learning_rate_node=tf.placeholder(tf.float32)
        #    =0.0003/VALUE_NET_BATCH_SIZE
        opt_op=tf.train.AdamOptimizer(learning_rate_node).minimize(mse_train)
        #opt_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_train)
        train_mse_summary=tf.scalar_summary("mse_train", mse_train)

        tf.get_variable_scope().reuse_variables()
        eval_output = self.model(eval_states_node, keep_prob_node=keep_prob_node)
        mse_eval=tf.reduce_mean(tf.square(tf.sub(eval_output, eval_targets_node)))
        test_mse_summary=tf.scalar_summary("mse_test", mse_eval)

        overall_train_error=tf.placeholder(tf.float32)
        overall_train_error_summary=tf.scalar_summary("overall_mse_train", overall_train_error)
        sess=tf.Session()
        sess.run(tf.initialize_all_variables())

        #var_dict = {self.input_layer.weight.op.name: self.input_layer.weight,
        #            self.input_layer.bias.op.name: self.input_layer.bias}

        #for i in xrange(self.num_hidden_layer):
        #   var_dict[self.conv_layer[i].weight.op.name] = self.conv_layer[i].weight
        #    var_dict[self.conv_layer[i].bias.op.name] = self.conv_layer[i].bias
        #saver = tf.train.Saver(var_list=var_dict)
        #sl_model = os.path.join(MODELS_DIR, SLMODEL_NAME)
        #restore variables from SL model
        #saver.restore(sess, sl_model)

        saver=tf.train.Saver()
        ckpt=tf.train.get_checkpoint_state(VALUE_NET_MODEL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        train_writer=tf.train.SummaryWriter(FLAGS.summaries_dir+"/train", sess.graph)
        test_writer=tf.train.SummaryWriter(FLAGS.summaries_dir+"/test")

        gl_step=0
        step=0
        if not os.path.exists(os.path.dirname(VALUE_NET_MODEL_PATH)):
            print("creating valuenet model directory")
            os.mkdir(os.path.dirname(VALUE_NET_MODEL_PATH))

        last_epoch_step=0
        sum_train_error=0.0
        lr=FLAGS.alpha/VALUE_NET_BATCH_SIZE
        while epoch_count < num_epoch:
            offset, next_epoch=self.prepare_batch(train_raw_states, offset, self.batch_states, self.batch_label,
                                                  batchsize=VALUE_NET_BATCH_SIZE)
            if next_epoch:
                epoch_count +=1
            if (next_epoch or step==0):
                self.prepare_batch(eval_raw_states, 0, self.eval_batch_states, self.eval_batch_label, EVAL_SIZE)
                eval_error, summary = sess.run([mse_eval, test_mse_summary], feed_dict={eval_states_node: self.eval_batch_states,
                                                                eval_targets_node: self.eval_batch_label, keep_prob_node:1.0, learning_rate_node:lr})
                print("Test Mean Square Error:", eval_error)
                saver.save(sess, VALUE_NET_MODEL_PATH, global_step=gl_step)
                gl_step += 1
                test_writer.add_summary(summary, epoch_count)

                if step > 0:
                    count=step-last_epoch_step
                    ave=sum_train_error/count
                    print("Training Mean Square Error:", ave)
                    overall_train_error_v=sess.run(overall_train_error_summary,
                                                 feed_dict={overall_train_error:ave})
                    test_writer.add_summary(overall_train_error_v, epoch_count)
                    sum_train_error = 0.0
                    if eval_error - ave > 0.1:
                        lr=lr/2.0
                last_epoch_step = step

            if (step+1) % 100 ==0:

                run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata=tf.RunMetadata()
                summary, train_error, _=sess.run([train_mse_summary, mse_train, opt_op],
                                                 feed_dict={batch_states_node:self.batch_states,
                                                            batch_targets_node:self.batch_label, keep_prob_node:0.5, learning_rate_node:lr},
                                                 options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%d'%step)
                train_writer.add_summary(summary, step)
                print("adding run metadata for", step)
                print("epoch ", epoch_count, "step ", step, "train error", train_error)
                sum_train_error += train_error
            else:

                train_error, summary,  _ = sess.run([mse_train, train_mse_summary, opt_op],
                                                    feed_dict={batch_states_node:self.batch_states,
                                                               batch_targets_node:self.batch_label, keep_prob_node:0.5, learning_rate_node:lr})
                train_writer.add_summary(summary, step)
                print("epoch ", epoch_count, "step ", step, "train error", train_error)
                sum_train_error += train_error
            step += 1

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

    def declare_layers(self, num_hidden_layer):
        self.num_hidden_layer = num_hidden_layer
        self.input_layer = Layer("input_layer", paddingMethod="VALID")
        self.conv_layer = [None] * num_hidden_layer
        for i in range(num_hidden_layer):
            self.conv_layer[i] = Layer("conv%d_layer" % i)

    # will reue this model for evaluation
    def model(self, data_node, keep_prob_node, kernal_size=(3, 3), kernal_depth=24):
        output = [None] * self.num_hidden_layer
        weightShape0 = kernal_size + (INPUT_DEPTH, kernal_depth)
        output[0] = self.input_layer.convolve(data_node, weight_shape=weightShape0, bias_shape=(kernal_depth,))

        weightShape = kernal_size + (kernal_depth, kernal_depth)
        for i in range(self.num_hidden_layer - 2):
            output[i + 1] = self.conv_layer[i].convolve(output[i], weight_shape=weightShape, bias_shape=(kernal_depth,))

        output[self.num_hidden_layer-1]=self.conv_layer[self.num_hidden_layer-2].\
            convolve_no_relu(output[self.num_hidden_layer-2], weight_shape=weightShape, bias_shape=(kernal_depth,))

        logits = self.conv_layer[self.num_hidden_layer - 1].move_logits(output[self.num_hidden_layer - 1], BOARD_SIZE, value_net=True)

        value_out_layer=Layer("value_output_layer", paddingMethod="VALID")
        v=value_out_layer.value_estimation(logits, 32, keep_prob_node)
        return v


def main(argv=None):
    vnet=ValueNet()
    train_path=tf.flags.FLAGS.train_example_path
    test_path=tf.flags.FLAGS.test_example_path

    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    vnet.train(train_path, test_path, num_epoch=FLAGS.n_epoches)

if __name__ == "__main__":
    tf.app.run()