import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNEL = 1
NUM_LABEL = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层
FC_SIZE = 512

#定义卷积神经网络的前向传播过程。
def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv'):
        w = tf.get_variable('w', [CONV1_SIZE,CONV1_SIZE,NUM_CHANNEL,CONV1_DEEP],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b',shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        #使用边长为5，深度为32的过滤器，过滤器移动步长为1，且使用全0填充。
        conv1 = tf.nn.conv2d(input_tensor,w,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,b))

    #实现第二程池化层的前向传播过程，最大池化，边长为2，使用全0填充
    with tf.variable_scope('layer2-pool'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #声明第三层卷积层。输入时上一层输出(14*14*32)
    with tf.variable_scope('layer3-conv'):
        w = tf.get_variable('w', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b',shape=[CONV2_DEEP],initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, w, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b))

    #实现第四层池化前向传播，输入为14*14*64，输出为7*7*64
    with tf.variable_scope('layer4-pool'):
        # pool2 size is [batch_size,7,7,64]
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


    # 接下来是全连接层，需要将pool2转换为一维向量，作为后面的输入
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2,[-1,nodes])
    # reshaped = tf.reshape(pool2,[BATCH_SIZE,-1])
    # print(reshaped.get_shape())

    #声明第六层全连接层，输入为512的向量，输出为一组长度为10的向量。
    with tf.variable_scope('layer5-fc1'):
        fc1_w = tf.get_variable('w',shape=[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        try:
            # 只有全连接层的权重需要加入正则化
            if regularizer != None:
                tf.add_to_collection('loss',regularizer(fc1_w))
        except:
            pass
        fc1_b = tf.get_variable('b',shape=[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w) + fc1_b)
        # 使用Dropout随机将部分节点的输出改为0，为了防止过拟合的现象，从而使模型在测试数据中表现更好。
        # dropout一般只会在全连接层使用。
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_w = tf.get_variable('w', shape=[FC_SIZE, NUM_LABEL], initializer=tf.truncated_normal_initializer(stddev=0.1))
        try:
            if regularizer != None:
                tf.add_to_collection('loss', regularizer(fc2_w))
        except:
            pass
        fc2_b = tf.get_variable('b', shape=[NUM_LABEL], initializer=tf.constant_initializer(0.1))
        # 最后一层的输出，不需要加入激活函数
        logit = tf.matmul(fc1, fc2_w) + fc2_b

    return logit