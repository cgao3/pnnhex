" this is a trash file for testing some apis, useless in real work"

import numpy as np
import tensorflow as tf

input_node = tf.placeholder(dtype=tf.float32, shape=(2,))

def model(test=False):
    with tf.variable_scope("aaaa") as sp:
        if test:
            sp.reuse_variables()
        z=tf.get_variable(name="zzz", initializer=tf.random_uniform_initializer(), shape=(1,))
        y=tf.mul(input_node, z)

    if test:
        with tf.variable_scope("z222") as sp2:
            z2=tf.get_variable(name="z2", initializer=tf.constant_initializer(3.0), shape=(1,))

        y=tf.add(y, z2)
    return z,y

x=np.ndarray(dtype=np.float32, shape=(2))
x.fill(2.0)
print("x=", x)
with tf.Session() as sess:
    z,y=model()
    print(y.name)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver2 = tf.train.Saver({"fdsafda/this_is_z":z})
    saver2.save(sess, "hello.ckpt")
    tt=sess.run(y, feed_dict={input_node: x})
    #tf.get_variable_scope().reuse_variables()
    print(z.op.name)
    print(tt, "\n")

if(True):
    with tf.Session() as sess2:
        z,yy=model(True)
        sess2.run(tf.initialize_all_variables())

        saver2.restore(sess2, "hello.ckpt")
        print("zname:")
        print(z.op.name)
        tt = sess2.run(yy, feed_dict={input_node: x})
        print(tt, "\n")




def test(x, y=3, z=4):
    print(x+y+z)

test(5,z=15)


# Andrew's model for alfuego
#
#layers = [ConvLayerSpec(num_filters=64, filter_shape=(5, 5)),
#          ConvLayerSpec(num_filters=64, filter_shape=(3, 3), pad='same'),
#          ConvLayerSpec(num_filters=64, filter_shape=(3, 3), pad='same'),
#          ConvLayerSpec(num_filters=32, filter_shape=(3, 3), pad='same'),
#          ConvLayerSpec(num_filters=32, filter_shape=(3, 3), pad='same'),
#          ConvLayerSpec(num_filters=16, filter_shape=(3, 3), pad='same'),
#          ConvLayerSpec(num_filters=16, filter_shape=(3, 3), pad='same'),
#          ConvLayerSpec(num_filters=1, filter_shape=(1, 1), pad='same')]
#
#data_node is a tf.plceholder with the shape (batch_size, width, height, depth)
from layer import Layer
def model(self, data_node):
    nd_layers = 7
    input_layer = Layer("input_layer", paddingMethod="VALID")
    conv_layer = [None] * nd_layers
    for i in range(nd_layers):
        self.conv_layer[i] = Layer("conv%d_layer" % i)

    input_depth=12
    weightShape0 = (5,5) + (input_depth, 64)
    output = input_layer.convolve(data_node, weight_shape=weightShape0, bias_shape=(64,))
    output = conv_layer[0].convolve(output, weight_shape=(3,3,64,64), bias_shape=(64,))
    output = conv_layer[1].convolve(output, weight_shape=(3,3,64,64), bias_shape=(64,))
    output = conv_layer[2].convolve(output, weight_shape=(3,3,64,32), bias_shape=(32,))
    output = conv_layer[3].convolve(output, weight_shape=(3,3,32,32), bias_shape=(32,))
    output = conv_layer[4].convolve(output, weight_shape=(3,3,32,16), bias_shape=(16,))
    output = conv_layer[5].convolve(output, weight_shape=(3,3,16,16), bias_shape=(16,))
    #output = conv_layer[6].convolve(output, weight_shape=(1,1,16,1), bias_shape=(1,))
    boardsize=7
    logits = conv_layer[6].move_logits(output, boardsize)

    #another layer is softmax on this logits
    return logits
