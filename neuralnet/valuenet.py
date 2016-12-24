from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from six.moves import xrange

import numpy as np
import tensorflow as tf
from neuralnet.layer import Layer
import os
from utils.read_data import *
from neuralnet.supervised import MODELS_DIR


VALUE_NET_MODEL_NAME="value_model.ckpt"
tf.flags.DEFINE_string("summaries_dir2","/tmp/valuenet_logs", "where the summaries are")
tf.flags.DEFINE_integer("nSteps2", 500000, "number of steps to train value net")
tf.flags.DEFINE_float("learing_rate", 0.001, "value of learning rate")

FLAGS=tf.app.flags.FLAGS
class ValueNet2(object):
    def __init__(self, srcTrainDataPath, srcTestDataPath, srcTestPathFinal=None):
        self.srcTrainPath = srcTrainDataPath
        self.srcTestPath = srcTestDataPath
        self.srcTestPathFinal = srcTestPathFinal

    def _setup_architecture(self, nLayers):
        self.nLayers = nLayers
        self.inputLayer = Layer("InputLayer", paddingMethod="VALID")
        self.convLayers = [Layer("ConvLayer%d" % i) for i in xrange(nLayers)]

    def model(self, dataNode, kernalSize=(3, 3), kernalDepth=64, numValueUnits=48):
        weightShape = kernalSize + (INPUT_DEPTH, kernalDepth)
        output = self.inputLayer.convolve(dataNode, weight_shape=weightShape, bias_shape=(kernalDepth,))

        weightShape = kernalSize + (kernalDepth, kernalDepth)
        for i in xrange(self.nLayers):
            out = self.convLayers[i].convolve(output, weight_shape=weightShape, bias_shape=(kernalDepth,))
            output = out
        logits = self.convLayers[self.nLayers - 1].move_logits(output, BOARD_SIZE, value_net=True)
        self.valueLayer=Layer("ValueOutputLayer", paddingMethod="VALID")
        value=self.valueLayer.value_estimation(logits, numValueUnits)
        return value

    def train(self, nSteps):
        self.batchInputNode = tf.placeholder(dtype=tf.float32,
                                             shape=(BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH),
                                             name="BatchTrainInputNode")
        self.batchLabelNode = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,), name="BatchTrainLabelNode")

        self.xInputNode = tf.placeholder(dtype=tf.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH),
                                         name="x_input_node")
        fake_input = np.ndarray(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        fake_input.fill(0)

        self._setup_architecture(nLayers=5)
        batchPredictedValue = self.model(self.batchInputNode)
        MSE=tf.reduce_mean(tf.square(tf.sub(batchPredictedValue, self.batchLabelNode)))
        opt = tf.train.AdamOptimizer().minimize(MSE)

        tf.get_variable_scope().reuse_variables()
        self.xLogits = self.model(self.xInputNode)

        trainDataUtil = ValueUtil(self.srcTrainPath, batch_size=BATCH_SIZE)
        testDataUtil = ValueUtil(self.srcTestPath, batch_size=BATCH_SIZE)

        msePlaceholder = tf.placeholder(tf.float32)
        mseTrainSummary = tf.summary.scalar("Mean Square Error (Training)", msePlaceholder)
        mseValidateSummary = tf.summary.scalar("Mean Square Error (Validating)", msePlaceholder)

        saver = tf.train.Saver()
        print_frequency = 20
        test_frequency = 50
        step = 0
        epoch_num = 0
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            print("Initialized all variables!")
            trainWriter = tf.summary.FileWriter(FLAGS.summaries_dir2 + "/" + repr(nSteps) + "/train", sess.graph)
            validateWriter = tf.summary.FileWriter(FLAGS.summaries_dir2 + "/" + repr(nSteps) + "/validate", sess.graph)
            # trainWriter = tf.train.SummaryWriter(FLAGS.summaries_dir + "/train", sess.graph)
            # validateWriter = tf.train.SummaryWriter(FLAGS.summaries_dir + "/validate", sess.graph)
            while step < nSteps:
                nextEpoch = trainDataUtil.prepare_batch()
                if nextEpoch: epoch_num += 1
                inputs = trainDataUtil.batch_positions.astype(np.float32)
                labels = trainDataUtil.batch_labels.astype(np.float32)
                feed_dictionary = {self.batchInputNode: inputs, self.batchLabelNode: labels}
                _, run_error = sess.run([opt, MSE], feed_dict=feed_dictionary)

                if step % print_frequency:
                    print("epoch: ", epoch_num, "step:", step, "MSE:", run_error)
                    summary = sess.run(mseTrainSummary, feed_dict={msePlaceholder: run_error})
                    trainWriter.add_summary(summary, step)
                if step % test_frequency == 0:
                    hasOneEpoch = False
                    sum_run_error = 0.0
                    ite = 0
                    while hasOneEpoch == False:
                        hasOneEpoch = testDataUtil.prepare_batch()
                        x_input = testDataUtil.batch_positions.astype(np.float32)
                        feed_d = {self.batchInputNode: x_input, self.batchLabelNode:testDataUtil.batch_labels}
                        run_error = sess.run(MSE, feed_dict=feed_d)
                        sum_run_error += run_error
                        ite += 1
                    run_error = sum_run_error / ite
                    print("Validation MSE", run_error)
                    summary = sess.run(mseValidateSummary, feed_dict={msePlaceholder: run_error})
                    validateWriter.add_summary(summary, step)

                step += 1
            print("saving value net computation graph for c++ inference")
            sl_model_dir = os.path.dirname(MODELS_DIR)
            tf.train.write_graph(sess.graph_def, sl_model_dir, "valuegraph.pbtxt")
            tf.train.write_graph(sess.graph_def, sl_model_dir, "valuegraph.pb", as_text=False)
            saver.save(sess, os.path.join(sl_model_dir, VALUE_NET_MODEL_NAME), global_step=step)

            print("Testing error on test data is:")
            testDataUtil.close_file()
            testDataUtil = ValueUtil(self.srcTestPathFinal, batch_size=BATCH_SIZE)
            hasOneEpoch = False
            sum_run_error = 0.0
            ite = 0
            while hasOneEpoch == False:
                hasOneEpoch = testDataUtil.prepare_batch()
                x_input = testDataUtil.batch_positions.astype(np.float32)
                feed_d = {self.batchInputNode: x_input, self.batchLabelNode:testDataUtil.batch_labels}
                run_error = sess.run(MSE, feed_dict=feed_d)
                sum_run_error += run_error
                ite += 1
            print("Testing MSE is:", sum_run_error / ite)
            sess.run(self.xLogits, feed_dict={self.xInputNode: fake_input})
        trainDataUtil.close_file()
        testDataUtil.close_file()


def main(argv=None):
    vnet=ValueNet2(srcTrainDataPath="storage/position-value/8x8/train.txt",
                   srcTestDataPath="storage/position-value/8x8/validate.txt",
                   srcTestPathFinal="storage/position-value/8x8/test.txt")

    if tf.gfile.Exists(FLAGS.summaries_dir2):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir2)
    tf.gfile.MakeDirs(FLAGS.summaries_dir2)

    vnet.train(FLAGS.nSteps2)

if __name__ == "__main__":
    tf.app.run()