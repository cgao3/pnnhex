from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import time
import numpy as np

from neuralnet.layer import Layer
from utils.read_data import *
from six.moves import xrange;
import os

MODELS_DIR="models/"
SLMODEL_NAME="slmodel.ckpt"


tf.flags.DEFINE_string("summaries_dir","/tmp/slnet_logs", "where the summaries are")
tf.app.flags.DEFINE_integer("nSteps", 40001, "number of training steps")
tf.app.flags.DEFINE_boolean("inference", False, "for inference?")
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

    def train(self, nSteps):
        self.batchInputNode = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH),name="BatchTrainInputNode")
        self.batchLabelNode = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,), name="BatchTrainLabelNode")

        self.xInputNode=tf.placeholder(dtype=tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), name="x_input_node")
        fake_input=np.ndarray(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        fake_input.fill(0)

        self.setup_architecture(nLayers=7)
        batchLogits=self.model(self.batchInputNode)
        batchPrediction=tf.nn.softmax(batchLogits)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(batchLogits, self.batchLabelNode))
        opt=tf.train.AdamOptimizer().minimize(loss)

        tf.get_variable_scope().reuse_variables()
        self.xLogits=self.model(self.xInputNode)
        
        trainDataUtil = PositionUtil9(positiondata_filename=self.srcTrainPath, batch_size=BATCH_SIZE)
        trainDataUtil.enableRandomFlip=True
        testDataUtil = PositionUtil9(positiondata_filename=self.srcTestPath, batch_size=BATCH_SIZE)

        accuracyPlaceholder = tf.placeholder(tf.float32)
        accuracyTrainSummary = tf.summary.scalar("Accuracy (Training)", accuracyPlaceholder)
        accuracyValidateSummary = tf.summary.scalar("Accuracy (Validating)", accuracyPlaceholder)

        saver=tf.train.Saver(max_to_keep=10)
        print_frequency=20
        test_frequency=500
        save_frequency=5000
        step=0
        epoch_num=0

        BEST_ERROR=100.0

        with tf.Session() as sess:
            init=tf.variables_initializer(tf.global_variables(), name="init_node")
            sess.run(init)
            print("Initialized all variables!")
            trainWriter = tf.summary.FileWriter(FLAGS.summaries_dir+"/"+repr(nSteps)+"/train", sess.graph)
            validateWriter = tf.summary.FileWriter(FLAGS.summaries_dir +"/"+repr(nSteps)+ "/validate", sess.graph)

            sl_model_dir = os.path.dirname(MODELS_DIR)
            while step < nSteps:
                nextEpoch=trainDataUtil.prepare_batch()
                if nextEpoch: epoch_num += 1
                inputs=trainDataUtil.batch_positions.astype(np.float32)
                labels=trainDataUtil.batch_labels.astype(np.uint16)
                feed_dictionary={self.batchInputNode:inputs, self.batchLabelNode:labels}
                _, run_loss=sess.run([opt, loss], feed_dict=feed_dictionary)

                if step % print_frequency:
                    run_predict=sess.run(batchPrediction, feed_dict={self.batchInputNode:inputs})
                    run_error=error_rate(run_predict,trainDataUtil.batch_labels)
                    print("epoch: ", epoch_num, "step:", step, "loss:", run_loss, "error_rate:", run_error )
                    summary = sess.run(accuracyTrainSummary, feed_dict={accuracyPlaceholder: 100.0-run_error})
                    trainWriter.add_summary(summary, step)
                if step % test_frequency == 0:
                    hasOneEpoch=False
                    sum_run_error=0.0
                    ite=0
                    while hasOneEpoch==False:
                        hasOneEpoch=testDataUtil.prepare_batch()
                        x_input = testDataUtil.batch_positions.astype(np.float32)
                        feed_d = {self.batchInputNode: x_input}
                        predict = sess.run(batchPrediction, feed_dict=feed_d)
                        run_error = error_rate(predict, testDataUtil.batch_labels)
                        sum_run_error += run_error
                        ite +=1
                    run_error=sum_run_error/ite
                    print("evaluation error rate", run_error)
                    summary = sess.run(accuracyValidateSummary, feed_dict={accuracyPlaceholder: 100.0-run_error})
                    validateWriter.add_summary(summary, step)
                if step>=40000 and step %save_frequency==0:
                    testDataUtil_save = PositionUtil9(positiondata_filename=self.srcTestPathFinal, batch_size=BATCH_SIZE)
                    hasOneEpoch = False
                    sum_run_error = 0.0
                    ite = 0
                    while hasOneEpoch == False:
                        hasOneEpoch = testDataUtil_save.prepare_batch()
                        x_input = testDataUtil_save.batch_positions.astype(np.float32)
                        feed_d = {self.batchInputNode: x_input}
                        predict = sess.run(batchPrediction, feed_dict=feed_d)
                        run_error = error_rate(predict, testDataUtil_save.batch_labels)
                        sum_run_error += run_error
                        ite += 1
                    run_error = sum_run_error / ite
                    print("Test evaluation error rate", run_error)
                    if run_error < BEST_ERROR:
                        print("smaller than best error: ", BEST_ERROR, "; saving the model")
                        saver.save(sess, os.path.join(sl_model_dir, SLMODEL_NAME), global_step=step)
                        BEST_ERROR = run_error
                    testDataUtil_save.close_file()
                step += 1
            print("saving computation graph for c++ inference")
            tf.train.write_graph(sess.graph_def, sl_model_dir, "graph.pbtxt")
            tf.train.write_graph(sess.graph_def, sl_model_dir, "graph.pb", as_text=False)
            saver.save(sess, os.path.join(sl_model_dir, SLMODEL_NAME), global_step=step)

            print("Testing error on test data is:")
            testDataUtil.close_file()
            testDataUtil=PositionUtil9(positiondata_filename=self.srcTestPathFinal, batch_size=BATCH_SIZE)
            hasOneEpoch=False
            sum_run_error=0.0
            sum2=0.0
            ite=0
            KValue=3
            while hasOneEpoch==False:
                hasOneEpoch = testDataUtil.prepare_batch()
                x_input = testDataUtil.batch_positions.astype(np.float32)
                feed_d = {self.batchInputNode: x_input}
                predict = sess.run(batchPrediction, feed_dict=feed_d)
                run_error = error_rate(predict, testDataUtil.batch_labels)
                top_k_run_error= error_topk(predict, testDataUtil.batch_labels, k=KValue)
                sum_run_error += run_error
                sum2 +=top_k_run_error
                ite += 1
            print("Testing error is:", sum_run_error/ite)
            writeout=open("test_error.txt","w")
            writeout.write("Testing error is: "+repr(sum_run_error/ite)+'\n')
            writeout.write("Top "+ repr(KValue)+" error: "+repr(sum2/ite))
            writeout.close()
            sess.run(self.xLogits, feed_dict={self.xInputNode:fake_input}) 
        trainDataUtil.close_file()
        testDataUtil.close_file()

def error_rate(predictions, labels):
    return 100.0 - 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]

def error_topk(predictions, labels, k):
    sortedIndArray=np.argsort(predictions, 1)[:,-k:]
    correctArray=[e for ind, e in enumerate(labels) if labels[ind] in sortedIndArray[ind]]
    return 100.0-100.0*len(correctArray)/predictions.shape[0]

def main(argv=None):
    if FLAGS.inference:
        slnet=SupervisedNet(srcTestDataPath=None, srcTrainDataPath=None, srcTestPathFinal=None)
        slnet.inference(FLAGS.slmodel_path)
        return

    slnet=SupervisedNet(srcTrainDataPath="storage/position-action/13x13/Train.txt",
                       srcTestDataPath="storage/position-action/13x13/Validate.txt",
                       srcTestPathFinal="storage/position-action/13x13/Test.txt")
    slnet.train(FLAGS.nSteps)

if __name__ == "__main__":
    tf.app.run()

