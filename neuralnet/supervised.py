from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import time
import numpy as np

from neuralnet.layer import Layer
from utils.read_data import *

import os

MODELS_DIR="models/"
SLMODEL_NAME="slmodel.ckpt"

tf.flags.DEFINE_string("summaries_dir","./train_logs/", "where the summaries are")
tf.app.flags.DEFINE_integer("nSteps", 500, "number of training steps")
tf.app.flags.DEFINE_boolean("test", False, "for inference?")
tf.app.flags.DEFINE_integer('n_hidden_layer', 8, 'number of hidden layers >=2')
tf.app.flags.DEFINE_string("checkpoint_path", '', 'location of the checkpoint to test')
tf.app.flags.DEFINE_string("topk", 1, 'top k for test')

tf.app.flags.DEFINE_string("slmodel_path", MODELS_DIR+SLMODEL_NAME, "for inference, please indicate SL Model path.")

FLAGS = tf.app.flags.FLAGS

'''game positions supervised learning'''
class SupervisedNet(object):

    def __init__(self, srcTrainDataPath, srcTestDataPath, srcTestPathFinal=None):
        self.srcTrainPath=srcTrainDataPath
        self.srcTestPath=srcTestDataPath
        self.srcTestPathFinal=srcTestPathFinal

    def setup_architecture(self, nLayers):
        self.nLayers=nLayers
        self.inputLayer=Layer("InputLayer", paddingMethod="VALID")
        self.convLayers=[Layer("ConvLayer%d"%i) for i in xrange(nLayers)]

    def model(self, dataNode, kernalSize=(3,3), kernalDepth=128):
        #first conv uses 5x5 filter,
        if PADDINGS >1:
            assert(PADDINGS==2)
            #check if the padding border if
            weightShape=(5,5)+(INPUT_DEPTH, kernalDepth)
        else:
            weightShape = kernalSize + (INPUT_DEPTH, kernalDepth)
        output=self.inputLayer.convolve(dataNode, weight_shape=weightShape, bias_shape=(kernalDepth,))

        #all other layers use 3x3
        weightShape=kernalSize+(kernalDepth, kernalDepth)
        for i in xrange(self.nLayers):
            out=self.convLayers[i].convolve(output, weight_shape=weightShape, bias_shape=(kernalDepth,))
            output=out
        logits=self.convLayers[self.nLayers-1].move_logits(output, BOARD_SIZE)
        return logits

    def test(self, checkpoint, topk=1):
        self.batchInputNode = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), name="x_input_node")
        self.batchLabelNode = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,), name="BatchTrainLabelNode")
        self.setup_architecture(nLayers=FLAGS.n_hidden_layer-1)
        batch_logits=self.model(self.batchInputNode)
        accuracy_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=self.batchLabelNode, k=topk), tf.float32))
        testDataUtil = PositionUtil9(positiondata_filename=self.srcTestPath, batch_size=BATCH_SIZE)
        testDataUtil.enableRandomFlip=False
        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            batch_no = 0
            over_all_acc = 0.0
            while True:
                is_next_epoch = testDataUtil.prepare_batch()
                feed_diction={self.batchInputNode:testDataUtil.batch_positions,
                              self.batchLabelNode:testDataUtil.batch_labels}
                acc = sess.run(accuracy_op, feed_dict=feed_diction)
                print("batch no.: ", batch_no, " test accuracy: ", acc)
                batch_no += 1
                over_all_acc += acc
                if is_next_epoch:
                    break
            print("top: ", topk, " overall accuracy on test dataset", self.srcTestPath, " is ", over_all_acc / batch_no)
            testDataUtil.close_file()

    def train(self, nSteps):
        self.batchInputNode = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH),name="x_input_node")
        self.batchLabelNode = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,), name="BatchTrainLabelNode")

        print('n_hidden_layer=', FLAGS.n_hidden_layer)
        self.setup_architecture(nLayers=FLAGS.n_hidden_layer-1)
        batchLogits=self.model(self.batchInputNode)
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(self.batchLabelNode, tf.cast(tf.argmax(batchLogits, 1), tf.int32)), tf.float32))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=batchLogits, labels=self.batchLabelNode))
        opt=tf.train.AdamOptimizer().minimize(loss)

        trainDataUtil = PositionUtil9(positiondata_filename=self.srcTrainPath, batch_size=BATCH_SIZE)
        trainDataUtil.enableRandomFlip=True


        accuracy_writer=open(FLAGS.summaries_dir+"/train_accuracy.txt", "w")
        loss_writer=open(FLAGS.summaries_dir+"/train_loss.txt", "w")
        saver=tf.train.Saver(max_to_keep=30)

        step=0
        epoch_num=0

        epoch_acc_sum=0.0
        eval_step=0
        with tf.Session() as sess:
            init=tf.variables_initializer(tf.global_variables(), name="init_node")
            sess.run(init)
            print("Initialized all variables!")

            sl_model_dir = os.path.dirname(MODELS_DIR)
            while step < nSteps:
                nextEpoch=trainDataUtil.prepare_batch()
                inputs=trainDataUtil.batch_positions.astype(np.float32)
                labels=trainDataUtil.batch_labels.astype(np.int32)
                feed_dictionary={self.batchInputNode:inputs, self.batchLabelNode:labels}
                run_loss=sess.run(loss, feed_dict=feed_dictionary)
                loss_writer.write('step:'+repr(step)+', loss: '+repr(run_loss)+'\n')

                if step % 50 == 0:
                   eval_step +=1
                   acc_train=sess.run(accuracy_op, feed_dict=feed_dictionary)
                   print("epoch: ", epoch_num, "step:", step, "loss:", run_loss, "accuracy", acc_train)
                   accuracy_writer.write(repr(step)+' '+repr(acc_train)+'\n')
                   epoch_acc_sum +=acc_train

                if nextEpoch:
                    epoch_num +=1
                    print('epoch num',epoch_num, 'epoch train acc:', epoch_acc_sum/eval_step)
                    accuracy_writer.write('epoch: ' + repr(epoch_num) + ' epoch acc: '+repr(epoch_acc_sum/eval_step) +'\n')
                    eval_step=0
                    epoch_acc_sum=0.0
                    saver.save(sess, os.path.join(FLAGS.summaries_dir+'/cnn_model.ckpt'), global_step=epoch_num)
                sess.run(opt, feed_dict=feed_dictionary)
                step += 1

            print("saving computation graph for c++ inference")
            tf.train.write_graph(sess.graph_def, FLAGS.summaries_dir, "graph.pbtxt")
            tf.train.write_graph(sess.graph_def, FLAGS.summaries_dir, "graph.pb", as_text=False)
            saver.save(sess, FLAGS.summaries_dir+'/cnn_model.ckpt', global_step=step)
        trainDataUtil.close_file()
        accuracy_writer.close()
        accuracy_writer.close()
        loss_writer.close()

def main(argv=None):

    slnet=SupervisedNet(srcTrainDataPath="storage/position-action/13x13/Train.txt",
                       srcTestDataPath="storage/position-action/13x13/Test.txt",
                       srcTestPathFinal="storage/position-action/13x13/Test.txt")
    if FLAGS.test:
        slnet.test(checkpoint=FLAGS.checkpoint_path, topk=FLAGS.topk)
        exit(0)
    slnet.train(FLAGS.nSteps)

if __name__ == "__main__":
    tf.app.run()

