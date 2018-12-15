import tensorflow as tf
#定义一个变量用于计算滑动平均，变量初始值为0.
v1 = tf.Variable(0,dtype=tf.float32)
#这里step变量模拟神经网络中的迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0,trainable=False)

#定义一个滑动平均的类,初始化给定衰减率0.99
ema = tf.train.ExponentialMovingAverage(0.99,step)
#定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个操作，这个列表的变量都会被更新
maintain_average_op = ema.apply([v1])
with tf.Session() as sess:
    #初始化所有变量。
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    #通过ema.average(v1)获取滑动平均之后变量的取值，在初始化之后变量v1的值和v1的滑动平均都为0
    print(sess.run([v1,ema.average(v1)]))

    #更新变量v1的值到5
    sess.run(tf.assign(v1,5))
    #更新v1的滑动平均值。衰减率为min{0.99,(1+step)/(10+step) = 0.1}=0.1,滑动平均值会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

    #更新step为10000
    sess.run(tf.assign(step,10000))
    #更新v1的值为10
    sess.run(tf.assign(v1,10))

    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))