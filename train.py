import tensorflow as tf


v1=tf.Variable(23.0)
v2=tf.Variable(2.0)

v3=tf.div(v1,v2)
v4=tf.mul(v1,v2)

init=tf.initialize_all_variables()

sess1=tf.Session()
sess1.run(init)
print(sess1.run(v3))

sess2=tf.Session()
sess2.run(init)
print(sess2.run(v4))
sess1.close()

sess2.close()