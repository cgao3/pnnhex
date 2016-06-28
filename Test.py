
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


def error_rate(predictions, labels):
    return 100.0 - 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]

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

logit=tf.matmul(data_x, W)+b

loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit, y))
opt=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
train_prediction=tf.nn.softmax(logit)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    _, predictions=sess.run([opt, train_prediction], feed_dict={data_x:X})
    print("error rate: ", error_rate(predictions=predictions, labels=y) )

one_x = tf.placeholder(dtype=tf.float32, shape=(1,D))
one_y = tf.placeholder(dtype=tf.int32, shape=(1,))

logit2=tf.matmul(one_x, W)+b
loss2=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit2, one_y))
opt2=tf.train.AdagradOptimizer(0.4)

grad_vars=opt2.compute_gradients(loss2)
grad_placeholder=[(tf.placeholder(dtype=tf.float32, shape=grad[1].get_shape()), grad[1]) for grad in grad_vars]
apply_grad_placeholder_op=opt2.apply_gradients(grad_placeholder)

opt2_op=opt2.minimize(loss2)
train_prediction2=tf.nn.softmax(logit2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sum_error=0.0
    grad_vals=[]
    for i in range(N*K):
        grad_val=sess.run([grad[0] for grad in grad_vars], feed_dict={one_x:X[i:i+1], one_y:y[i:i+1]})
        grad_vals.append(grad_val)

    for k in xrange(N*K):
        feed_diction = {}
        for i in range(len(grad_placeholder)):
            feed_diction[grad_placeholder[i][0]]=grad_vals[k][i]
        sess.run(apply_grad_placeholder_op,feed_dict=feed_diction)
    pred=sess.run(logit, feed_dict={data_x:X})
    print("error rate is ", error_rate(pred, y))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sum_error=0.0
    for i in range(K*N):
        _, pred=sess.run([opt2_op, train_prediction2], feed_dict={one_x:X[i:i+1], one_y:y[i:i+1]})

    pred=sess.run(logit, feed_dict={data_x:X})
    print("overall error", error_rate(pred,y))

