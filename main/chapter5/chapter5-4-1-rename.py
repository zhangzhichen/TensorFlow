import tensorflow as tf
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="other-v1")
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name="other-v2")

#声明tf.train.Saver用于保存模型
saver = tf.train.Saver()