import tensorflow as tf

v1 = tf.constant([1.0,2.0,3.0,4.0])
v2 = tf.constant([4.0,3.0,2.0,1.0])
sess = tf.InteractiveSession()
#比较每一个元素的大小
print(tf.greater(v1,v2).eval())
#返回最大的数(greater为true,选第一个，否则，第二个)
print(tf.where(tf.greater(v1,v2),v1,v2).eval())