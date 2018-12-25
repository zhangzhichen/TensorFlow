import tensorflow as tf
v = tf.Variable(0,dtype=tf.float32,name='v')
#在没有声明滑动平均模型时只有一个变量v，所以以下语句会输出“v:0”
for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.global_variables())
#在申明滑动平均模型后，tensorflow会自动生成一个影子变量
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v,10))
    sess.run(maintain_average_op)

    saver.save(sess,"D:\saver\chapter5\modelEMA.ckpt")
    print(sess.run([v,ema.average(v)]))