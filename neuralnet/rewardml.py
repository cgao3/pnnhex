from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import time
import numpy as np

from neuralnet.layer import Layer
from utils.positionutil import *
from six.moves import xrange;
import os

MODELS_DIR="models/rmlmodel/"
RMLMODEL_NAME= "rmlmodel.ckpt"


tf.flags.DEFINE_string("rmlsummaries_dir","/tmp/rmlnet_logs", "where the summaries are")
tf.app.flags.DEFINE_integer("nSteps3", 40001, "number of training steps")
tf.app.flags.DEFINE_boolean("inferencerml", False, "for inference?")
tf.app.flags.DEFINE_string("rmlmodel_path", MODELS_DIR + RMLMODEL_NAME, "for inference, please indicate SL Model path.")
FLAGS = tf.app.flags.FLAGS

'''game position,next move, value supervised learning.
Reward augmented machine learning'''
class SupervisedRMLNet(object):

    def __init__(self, srcTrainDataPath, srcTestDataPath, srcTestPathFinal=None):
        self.srcTrainPath=srcTrainDataPath
        self.srcTestPath=srcTestDataPath
        self.srcTestPathFinal=srcTestPathFinal

    def setup_architecture(self, nLayers):
        self.nLayers=nLayers
        self.inputLayer=Layer("InputLayer", paddingMethod="VALID")
        self.convLayers=[Layer("ConvLayer%d"%i) for i in xrange(nLayers)]

    def model(self, dataNode, kernalSize=(3,3), kernalDepth=48):
        weightShape=kernalSize+(INPUT_DEPTH, kernalDepth)
        output=self.inputLayer.convolve(dataNode, weight_shape=weightShape, bias_shape=(kernalDepth,))

        weightShape=kernalSize+(kernalDepth, kernalDepth)
        for i in xrange(self.nLayers):
            out=self.convLayers[i].convolve(output, weight_shape=weightShape, bias_shape=(kernalDepth,))
            output=out
        logits=self.convLayers[self.nLayers-1].move_logits(output, BOARD_SIZE)
        return logits

    def train(self, nSteps):
        tau=0.95
        self.batchInputNode = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH),name="BatchTrainInputNode")
        self.batchLabelNode = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,), name="BatchTrainLabelNode")

        self.xInputNode=tf.placeholder(dtype=tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), name="x_input_node")
        fake_input=np.ndarray(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        fake_input.fill(0)

        self.setup_architecture(nLayers=5)
        batchLogits=self.model(self.batchInputNode)
        batchPrediction=tf.nn.softmax(batchLogits)
        sum_loss=0.0
        for i in range(BATCH_SIZE):
            sum_loss =sum_loss - tf.log(batchPrediction[i][self.batchLabelNode[i]])

        loss=sum_loss/BATCH_SIZE
        opt=tf.train.AdamOptimizer().minimize(loss)

        tf.get_variable_scope().reuse_variables()
        self.xLogits=self.model(self.xInputNode)
        
        trainDataUtil = PositionUtilReward(positiondata_filename=self.srcTrainPath, batch_size=BATCH_SIZE)
        testDataUtil = PositionUtilReward(positiondata_filename=self.srcTestPath, batch_size=BATCH_SIZE, forTest=True)

        accuracyPlaceholder = tf.placeholder(tf.float32)
        accuracyTrainSummary = tf.summary.scalar("Accuracy (Training)", accuracyPlaceholder)
        accuracyValidateSummary = tf.summary.scalar("Accuracy (Validating)", accuracyPlaceholder)

        saver=tf.train.Saver(max_to_keep=10)
        print_frequency=20
        step=0
        epoch_num=0
        bestError=100.0
        bestTrainStep=None
        maxPatienceEpoch=10
        patience_begin=0

        with tf.Session() as sess:
            init=tf.variables_initializer(tf.global_variables(), name="init_node")
            sess.run(init)
            print("Initialized all variables!")
            trainWriter = tf.summary.FileWriter(FLAGS.rmlsummaries_dir+"/"+repr(nSteps)+"/train", sess.graph)
            validateWriter = tf.summary.FileWriter(FLAGS.rmlsummaries_dir +"/"+repr(nSteps)+ "/validate", sess.graph)

            rml_model_dir = os.path.dirname(MODELS_DIR)
            while step < nSteps:
                nextEpoch=trainDataUtil.prepare_batch()
                if nextEpoch:
                    epoch_num += 1
                    hasOneTestEpoch = False
                    sum_run_error = 0.0
                    ite = 0
                    while hasOneTestEpoch == False:
                        hasOneTestEpoch = testDataUtil.prepare_batch()
                        x_input = testDataUtil.batch_positions.astype(np.float32)
                        feed_d = {self.batchInputNode: x_input}
                        predict = sess.run(batchPrediction, feed_dict=feed_d)
                        run_error = errorRateTest(predict, testDataUtil.batch_labelSet)
                        sum_run_error += run_error
                        ite += 1
                    run_error = sum_run_error / ite
                    print("Epoch:", epoch_num, "Evaluation error rate", run_error)
                    summary = sess.run(accuracyValidateSummary, feed_dict={accuracyPlaceholder: 100.0 - run_error})
                    validateWriter.add_summary(summary, step)
                    saver.save(sess, os.path.join(rml_model_dir, RMLMODEL_NAME), global_step=step)
                    if(bestError>run_error):
                        bestError=run_error
                        bestTrainStep=step
                        patience_begin =0
                    else:
                        patience_begin +=1
                        if patience_begin >= maxPatienceEpoch:
                            break
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
                step += 1
            print("saving computation graph for c++ inference")
            tf.train.write_graph(sess.graph_def, rml_model_dir, "graph.pbtxt")
            tf.train.write_graph(sess.graph_def, rml_model_dir, "graph.pb", as_text=False)

            print("best Error is:", bestError, "best train step:", bestTrainStep)
            testDataUtil.close_file()
            testDataUtil=PositionUtilReward(positiondata_filename=self.srcTestPathFinal, batch_size=BATCH_SIZE, forTest=True)
            hasOneEpoch=False
            sum_run_error=0.0
            ite=0
            while hasOneEpoch==False:
                hasOneEpoch = testDataUtil.prepare_batch()
                x_input = testDataUtil.batch_positions.astype(np.float32)
                feed_d = {self.batchInputNode: x_input}
                predict = sess.run(batchPrediction, feed_dict=feed_d)
                run_error = errorRateTest(predict, testDataUtil.batch_labelSet)
                sum_run_error += run_error
                ite += 1
            print("Testing error is:", sum_run_error/ite)
            sess.run(self.xLogits, feed_dict={self.xInputNode:fake_input})
        trainDataUtil.close_file()
        testDataUtil.close_file()

def error_rate(predictions, labels):
    return 100.0 - 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]

def errorRateTest(batch_prediction, batch_labelSet):
    maxPredict=np.argmax(batch_prediction, 1)
    errorNum=0
    for i in range(len(maxPredict)):
        if maxPredict[i] not in batch_labelSet[i]:
            errorNum +=1

    return 100.0 * errorNum/len(maxPredict)


def errorRateTestTopK(batch_prediction, batch_labelSet, k=1):
    sortedIndArray = np.argsort(batch_prediction, 1)[:, -k:]
    errorNum=0
    for i in range(batch_prediction.shape[0]):
        isIn=False
        for label in batch_labelSet[i]:
            if label in sortedIndArray[i]:
                isIn=True
        if isIn==False:
            errorNum +=1

    return 100.0 * errorNum/batch_prediction.shape[0]

def main(argv=None):
    if FLAGS.inferencerml:
        slnet=SupervisedRMLNet(srcTestDataPath=None, srcTrainDataPath=None, srcTestPathFinal=None)
        slnet.inference(FLAGS.rmlmodel_path)
        return

    slnet=SupervisedRMLNet(srcTrainDataPath="storage/rml-data/8x8/Train.txt",
                       srcTestDataPath="storage/rml-data/8x8/Validate.txt",
                       srcTestPathFinal="storage/rml-data/8x8/Test.txt")
    slnet.train(FLAGS.nSteps3)

if __name__ == "__main__":
    tf.app.run()

