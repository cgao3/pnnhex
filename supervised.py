from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from layer import Layer
from read_data import *


input_layer = Layer("input_layer")

input_tensor = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, INPUT_CHANNEL_DEPTH), name="input_board_state")
input_labels = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, BOARD_SIZE * BOARD_SIZE))  # one-hot

output1 = input_layer.convolve(input_tensor, (3, 3, 3, 80), (80))

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
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, input_labels))

learning_rate = 0.01

opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print("Initialized!")
    print("loss looks ", loss)
    read_raw_data("data/train_games.dat")
    offset1, offset2 = 0, 0
    while(nEpoch < 1):
        off1, off2 = prepare_batch(offset1, offset2)
        x = batch_states.astype(np.float32)
        y = batch_labels.astype(np.float32)
        feed_diction = {input_tensor:x, input_labels:y}
        x = sess.run(opt, feed_dict=feed_diction)
        print("x ", x)
        print("loss", sess.run(loss))
        offset1, offset2 = off1, off2
        

