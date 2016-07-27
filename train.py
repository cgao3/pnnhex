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