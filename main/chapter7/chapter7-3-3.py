import tensorflow as tf
#申明先进先出队列，队列中最多100个元素，类型为实数
queue = tf.FIFOQueue(100,"float")
#定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([2])])

#使用tf.train.QueueRunner来创建多个线程运行队列的入队操作。
#tf.train.QueueRunner第一个参数给出了被操作的队列，[enqueue_op]*5表示需要启动5个线程，每隔线程运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)

#将定义过的queueRunner加入tensorflow计算图上指定的集合
tf.train.add_queue_runner(qr)
#定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    #使用tf.train.Corrdinator来协同启动的线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    #获取队列中的数值
    for _ in range(10):print(sess.run(out_tensor)[0])

    #使用tf.train.Coordinator来停止所有的线程。
    coord.request_stop()
    coord.join(threads)