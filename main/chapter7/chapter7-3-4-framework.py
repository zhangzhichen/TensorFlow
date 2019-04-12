import tensorflow as tf
#创建文件列表
files = tf.train.match_filenames_once("D:\picture/*")

#读取文件加入队列,将shuffle设置为false
filename_queue = tf.train.string_input_producer(files,shuffle=False)

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,features={
    'image':tf.FixedLenFeature([],tf.string),
    'label':tf.FixedLenFeature([],tf.int64),
    'height':tf.FixedLenFeature([],tf.int64),
    'width':tf.FixedLenFeature([],tf.int64),
    'channels':tf.FixedLenFeature([],tf.int64),
})
image,label = features['image'],features['label']
height,width = features['height'],features['width']
channels = features['channels']
#从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像。
decoded_image = tf.decode_raw(image,tf.uint8)
decoded_image.set_shape([height,width,channels])
#定义神经网络输入层图片的大小
image_size = 299
#preprocess_for_train预处理
distorted_image = preprocess_for_train(decoded_image,image_size,image_size,None)

#需要的batch
min_after_dequeue = 10000
batch_size = 1000
capacity = min_after_dequeue+3*batch_size
image_batch,label_batch = tf.train.shuffle_batch([distorted_image,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)

#定义神经网络的结果和优化过程，image_batch可以作为输入提供给神经网络的输入层.
#label_batch则提供了输入batch中样例的正确答案。
# 学习率
learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit,label_batch)
train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss)

#声明回话并运行神经网络的优化过程。
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # 神经网络训练过程。
    TRAINING_ROUNDS = 5000
    for i in range(TRAINING_ROUNDS):
        sess.run(train_step)

    #停止所有线程。
    coord.request_stop()
    coord.join(threads)