import tensorflow as tf

#创建一个先进先出队列，指定队列中最多可以保存两个元素，并指定类型为整数。
q = tf.FIFOQueue(2,"int32")
#使用enqueue_many函数来初始化队列中的元素，和变量初始化类似，在使用队列之前需要明确的调用这个初始化过程。
init = q.enqueue_many(([0,10],))
#使用dequeue函数将队列中第一个元素出队列。这个元素的值赋值给x
x = q.dequeue()
#将得到的值加1
y = x + 1
#重新放入队列
q_inc = q.enqueue([y])
with tf.Session() as sess:
    #初始化队列
    init.run()
    for _ in range(20):
        v,_ = sess.run([x,q_inc])
        print(v)