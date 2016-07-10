from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
from read_data import INPUT_DEPTH, INPUT_WIDTH, BOARD_SIZE
from policygradient import PGNetwork
from supervised import SLNetwork

SL_MODEL_PATH="savedModel/model.ckpt"
PG_MODEL_PATH="opponent_pool/model-10.ckpt"

class SLNetPlayer(object):

    def __init__(self):
        self.data_node=np.ndarray(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        slnet=SLNetwork("sl_player")
        with tf.name_scope("sl_player"):
            slnet.declare_layers()
            self.logits_node=slnet.model(self.data_node)

    def open_session(self):
        sess=tf.Session()
        self.sess=sess
        saver=tf.train.Saver()
        saver.restore(self.sess, SL_MODEL_PATH)

    def close_session(self):
        self.sess.close()

    def feedforward(self, input_tensor):
        logits = self.sess.run(self.logits_node, feed_dict={self.data_node: input_tensor})
        return logits


class PGNetPlayer(object):
    def __int__(self):
        self.data_node = np.ndarray(dtype=np.float32, shape=(1, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
        pgnet= PGNetwork("pg_player")
        with tf.name_scope("pg_player"):
            pgnet.declare_layers()
            self.logits_node=pgnet.model(self.data_node)


    def open_session(self):
        sess=tf.Session()
        self.sess=sess
        saver=tf.train.Saver()
        saver.restore(sess, PG_MODEL_PATH)


    def close_session(self):
        self.sess.close()

    def feedforward(self, input_tensor):
        logits=self.sess.run(self.logits_node, feed_dict={self.data_node: input_tensor})
        return logits


class ValueNet(object):

    def __init__(self):
        self.U=0
        self.slnet_player=SLNetPlayer()
        self.slnet_player.open_session()

        self.pgnet_player=PGNetPlayer()
        self.pgnet_player.open_session()


    def produce_data(self):
        pass

    def regression(self):
        pass

    def _random_move(self):
        pass

    def play(self):
        pass

    def close_sessions(self):
        self.pgnet_player.close_session()
        self.slnet_player.close_session()


def main(argv=None):
    vnet=ValueNet()

if __name__ == "__main__":
    tf.app.run()