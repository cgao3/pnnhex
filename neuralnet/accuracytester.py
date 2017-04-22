from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from utils.positionutil import *
from neuralnet.rewardml import *
from neuralnet.supervised import *
from utils.read_data import *
import argparse

class MovePredictionPredictor:
    def __init__(self,  srcIn, type="RML"):
        if type == "RML":
            self.putil=PositionUtilReward(positiondata_filename=srcIn, batch_size=BATCH_SIZE, forTest=True)
            self.net = SupervisedRMLNet(srcTestDataPath=None, srcTestPathFinal=None, srcTrainDataPath=None)
            self.net.setup_architecture(nLayers=8)

        else:
            self.putil=PositionUtil9(positiondata_filename=srcIn, batch_size=BATCH_SIZE)
            self.putil.enableRandomFlip=False
            self.net = SupervisedNet(srcTestDataPath=None, srcTestPathFinal=None, srcTrainDataPath=None)
            self.net.setup_architecture(nLayers=8)
        pass

    def inference(self, lastcheckpoint):
        self.xInputNode=tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), name="x_input_node")
        self.putil.prepare_batch()
        self.xLogits = self.net.model(self.xInputNode)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, lastcheckpoint)
            KTest=18
            for k in range(1,KTest,1):
                hasOneTestEpoch = False
                sum_run_error = 0.0
                ite = 0
                while hasOneTestEpoch == False:
                    hasOneTestEpoch = self.putil.prepare_batch()
                    x_input = self.putil.batch_positions.astype(np.float32)
                    feed_d = {self.xInputNode: x_input}
                    predict = sess.run(self.xLogits, feed_dict=feed_d)
                    if isinstance(self.putil, PositionUtilReward):
                        run_error = errorRateTestTopK(predict, self.putil.batch_labelSet, k=k)
                    else:
                        run_error = error_topk(predict, self.putil.batch_labels, k)
                    sum_run_error += run_error
                    ite += 1
                run_error = sum_run_error / ite
                print("Top", k, ", Accuracy: ", 100.0-run_error)
        self.putil.close_file()

class ValueMSETestor:
    def __init__(self):
        pass


if __name__ == "__main__":
    srcCheckPoint=""
    srcTestFile=""
    print("Testing Neural Net")
    parser = argparse.ArgumentParser(description='Neural Net Testor')
    parser.add_argument('model_path', type=str, help="the path of the model file")
    parser.add_argument('srcTestDataPath', type=str, help="the path to the test data")
    parser.add_argument('--value_net', action='store_true', help="value_net model?", default=False)
    args = parser.parse_args()
    if not args.value_net:
        print("Test Move prediction")
        testor=MovePredictionPredictor(srcIn=args.srcTestDataPath, type="ML")
        testor.inference(args.model_path)
    else:
        print("Test Value net")