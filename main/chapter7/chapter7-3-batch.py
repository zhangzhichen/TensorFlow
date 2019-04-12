import tensorflow as tf
#读取文件列表
files = tf.train.match_filenames_once("D:\saver/tfRecords/data.tfRecords-*")

#读取文件加入队列,将shuffle设置为false
filename_queue = tf.train.string_input_producer(files,shuffle=False)

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,features={'i':tf.FixedLenFeature([],tf.int64),'j':tf.FixedLenFeature([],tf.int64),})
example,label = features['i'],features['j']
#一个batch中样例的个数
batch_size = 3
#组合样例最多可以存储的样例个数
capacity = 1000+3*batch_size
#example_batch,label_batch = tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity = capacity)
#打乱顺序
example_batch,label_batch = tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity = capacity,min_after_dequeue=30)
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    #获取并打印组合之后的样例。在真实问题中，这个输出一般作为神经网络的输入
    for i in range(2):
        cur_example_batch,cur_label_batch = sess.run([example_batch,label_batch])
        print(cur_example_batch,cur_label_batch)

    coord.request_stop()
    coord.join(threads)