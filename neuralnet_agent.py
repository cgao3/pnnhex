import tensorflow as tf
import numpy as np

class network_agent(object):

    def __init__(self, model_location):
        self.model_path=model_location
        self.saver = tf.train.Saver()

    def load_model(self):
        saver=tf.train.Saver()
        self.sess=tf.Session()

    def play(self, input_tensor):

        return "a move"

    def close_all(self):
        self.sess.close()


