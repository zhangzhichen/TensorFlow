import tensorflow as tf

#过滤器定义，矩阵前两个参数是尺寸，第三个是当前层深度，第四个是过滤器深度
filter_weight = tf.get_variable('weights',[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=0.1))

biases = tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding='SAME')

#偏置项
bias = tf.nn.bias_add(conv,biases)

#relu激活函数去线性化
actived_conv = tf.nn.relu(bias)