import tensorflow as tf
#在命名空间为foo内创建名字为v的变量。
with tf.variable_scope("foo"):
    v = tf.get_variable("v",[1],initializer=tf.constant_initializer(1.0))

#这句代码会报错，因为已经有了v。
#with tf.variable_scope("foo"):
#    v = tf.get_variable("v",[1])

#reuse=True，只会取变量，如果不存在，会报错。
#reuse=False或者reuse=None，只会创建变量，如果存在，会报错。
with tf.variable_scope("foo",reuse=True):
    v1 = tf.get_variable("v",[1])
    print(v==v1)

with tf.variable_scope("bar",reuse=True):
    v1 = tf.get_variable("v",[1])
