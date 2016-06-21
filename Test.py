
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from layer import Layer
from read_data import *

num_epochs=10
train_step=10

tf.app.flags.DEFINE_boolean("training", True, "False if for evaluation")
tf.app.flags.DEFINE_string("check_point_dir","savedModel/", "path to save the model")
tf.app.flags.DEFINE_string("train_data_dir","data/","path to training data")
tf.app.flags.DEFINE_string("test_data_dir", "test/","path to test data")
FLAGS=tf.app.flags.FLAGS

def error_rate(predictions, labels):
    return 100.0- 100.0 * np.sum(np.argmax(predictions, 1) == labels)/predictions.shape[0]

def main(argv=None):

    train_data_node=tf.placeholder(tf.float32, shape=(BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH))
    train_labels_node=tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
    
    input_layer = Layer("input_layer", paddingMethod="VALID")
    
    output1 = input_layer.convolve(train_data_node, (3, 3, 3, 80), (80))
    
    conv1 = Layer("conv1_layer")
    out_conv1 = conv1.convolve(output1, weight_shape=(3, 3, 80, 80), bias_shape=(80))
    
    conv2 = Layer("conv2_layer")
    out_conv2 = conv2.convolve(out_conv1, weight_shape=(3, 3, 80, 80), bias_shape=(80))
    
    conv3 = Layer("conv3_layer")
    out_conv3 = conv3.convolve(out_conv2, weight_shape=(3, 3, 80, 80), bias_shape=(80))
    
    conv4 = Layer("conv4_layer")
    out_conv4 = conv4.convolve(out_conv3, weight_shape=(3, 3, 80, 80), bias_shape=(80))
    
    conv5 = Layer("conv5_layer")
    out_conv5 = conv5.convolve(out_conv4, weight_shape=(3, 3, 80, 80), bias_shape=(80))
    
    conv6 = Layer("conv6_layer")
    out_conv6 = conv6.convolve(out_conv5, weight_shape=(3, 3, 80, 80), bias_shape=(80))
    
    conv7 = Layer("conv7_layer")
    out_conv7 = conv7.convolve(out_conv6, weight_shape=(3, 3, 80, 80), bias_shape=(80))
    
    conv8 = Layer("conv8_layer")
    logits = conv8.one_filter_out(out_conv7, BOARD_SIZE)
    print("logits", logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels_node))

    train_prediction=tf.nn.softmax(logits)

    batch=tf.Variable(0)
    learning_rate=tf.train.exponential_decay(0.01,batch*BATCH_SIZE, train_step, 0.95,staircase=True)
    
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    saver=tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print("Initialized!")
        if not tf.app.flags.FLAGS.training:
            ckpt = tf.train.get_checkpoint_state(FLAGS.check_point_dir)
            if ckpt and ckpt.model_checkpoint_dir_path:
                print("restoring a model")
                saver.restore(sess,ckpt.model_checkpoint_dir_path)

        read_raw_data("data/train_games.dat")
        offset1, offset2 = 0, 0
        step=1
        training_step=10000
        nepoch = 0
        while(nepoch < num_epochs):
            off1, off2, nextepoch = prepare_batch(offset1, offset2)
            x = batch_states.astype(np.float32)
            y = batch_labels.astype(np.int32)
            feed_diction = {train_data_node:x, 
                            train_labels_node:y}
            _, loss_v, predictions=sess.run([opt,loss, train_prediction], feed_dict=feed_diction)
            print("epoch:", nepoch, "loss: ", loss_v, "error rate:", error_rate(predictions, batch_labels))
            offset1, offset2 = off1,off2
            if(nextepoch):
                nepoch += 1

        tf.save(sess,FLAGS.check_point_dir+"/model.ckpt")

            
if __name__ == "__main__":
    tf.app.run()

