import tensorflow as tf
weights = tf.constant([[1.0,-2.0],[-3.0,4.0]])
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    #会自动把L2的损失值除以2，使得求导结果更简洁
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))