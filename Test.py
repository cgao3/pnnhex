
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

N=100
D=2
K=3

X=np.zeros(shape=(N*K,D), dtype=np.float32)
y=np.zeros(shape=(N*K,), dtype=np.int32)

for j in xrange(K):
    ix=range(N*j, N*(j+1))
    r=np.linspace(0.0,1,N)
    t=np.linspace(j*4,(j+1)*4,N)+np.random.randn(N)*0.2
    X[ix]=np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix]=j

plt.scatter(X[:,0],X[:,1], c=y,s=40,cmap=plt.cm.Spectral)

W=tf.get_variable("Weight",shape=(D,K), initializer=tf.truncated_normal_initializer(stddev=0.1))
b=tf.get_variable("bias", shape=(K,), initializer=tf.constant_initializer(0.0))

data_x=tf.placeholder(dtype=tf.float32, shape=(N*K,D))
data_y=tf.placeholder(dtype=tf.int32, shape=(N*K, ))

one_x = tf.placeholder(dtype=tf.float32, shape=(1,D))
one_y = tf.placeholder(dtype=tf.int32, shape=(1,))

logit=tf.matmul(data_x, W)+b

logit2=tf.matmul(one_x, W)+b

R=tf.placeholder(dtype=tf.float32)

loss=tf.mul(R,tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit,data_y)))

label=tf.argmax(tf.nn.softmax(logit2),1)
loss2=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logit2, label[0:1]))

opt=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
opt2=tf.train.GradientDescentOptimizer(0.01)
grds_and_vars=opt2.compute_gradients(loss2)

grads=[grad for grad, _ in grds_and_vars]

grads_op=opt2.apply_gradients(grds_and_vars)

pred=tf.argmax(tf.nn.softmax(logit), 1)
correct_prediction = tf.equal(y, tf.cast(pred,tf.int32))
accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init=tf.initialize_all_variables()

sess=tf.Session()
tf.set_random_seed(10)
sess.run(init)
_ =sess.run([opt], feed_dict={R:0.3, data_x:X, data_y:y})
accu=sess.run(accuracy, feed_dict={data_x:X})
print("accuracy:", accu)
sess.close()

with tf.Session() as sess2:
    sess2.run(init)
    s=0.0
    for i in range(N*K):
        #sess2.run(opt2, feed_dict={one_x:X[i:i+1], one_y:y[i:i+1]})
        #sess2.run(grads_op, feed_dict={one_x:X[i:i+1], one_y:y[i:i+1]})
        sess2.run(grads_op, feed_dict={one_x:X[i:i+1]})

    bgrads=[]
    for i in range(N*K):
        pred2 = tf.argmax(tf.nn.softmax(logit2), 1)
        correct_pred=tf.equal(y[i:i+1], tf.cast(pred2, tf.int32))
        accu=tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acy=sess2.run(accu, feed_dict={one_x:X[i:i+1], one_y:y[i:i+1]})
        s += acy
        #print("accuracy is :", acy)
    print("overall accuracy: ", s/(N*K))

