from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from utils.positionutil import *
from neuralnet.rewardml import *
import argparse

class MovePredictionPredictor:
    def __init__(self):
        pass

    def inference(self, lastcheckpoint, srcTestDataInput):
        srcIn = srcTestDataInput
        self.xInputNode=tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), name="x_input_node")
        putil=PositionUtilReward(positiondata_filename=srcIn, batch_size=BATCH_SIZE, forTest=True)
        putil.prepare_batch()
        net= SupervisedRMLNet(srcTestDataPath=None, srcTestPathFinal=None, srcTrainDataPath=None)
        net.setup_architecture(nLayers=5)
        self.xLogits = net.model(self.xInputNode)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, lastcheckpoint)
            KTest=12
            for k in range(1,KTest,1):
                hasOneTestEpoch = False
                sum_run_error = 0.0
                ite = 0
                while hasOneTestEpoch == False:
                    hasOneTestEpoch = putil.prepare_batch()
                    x_input = putil.batch_positions.astype(np.float32)
                    feed_d = {self.xInputNode: x_input}
                    predict = sess.run(self.xLogits, feed_dict=feed_d)
                    run_error = errorRateTestTopK(predict, putil.batch_labelSet, k=k)
                    sum_run_error += run_error
                    ite += 1
                run_error = sum_run_error / ite
                print("Top", k, ", Accuracy: ", 100.0-run_error)
        putil.close_file()

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
        testor=MovePredictionPredictor()
        testor.inference(args.model_path, args.srcTestDataPath)
    else:
        print("Test Value net")