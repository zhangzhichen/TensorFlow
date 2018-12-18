import tensorflow as tf
import datetime
from tensorflow.examples.tutorials.mnist import input_data

#MNIST数据集相关的常数。
INPUT_NODE = 784  #输入层的节点数。等于图片的像素数量
OUTPUT_NODE = 10  #0到9

#配置神经网络的参数
LAYER1_NODE = 500 #隐藏层节点。只使用一个隐藏层，500个节点。

BATCH_SIZE = 100 #一个训练batch中训练数据个数，数字越小，越接近随机梯度下降。数字越大，越接近梯度下降

LEARNING_RATE_BASE = 0.8  #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化损失项在损失函数中的系数。
TRAINGING_STEPS = 3000   #训练轮数。
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值。
    if avg_class == None:
        #计算隐藏层的前向传播结果。使用ReLU激活函数。
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        #计算输出层的前向传播结果
        return tf.matmul(layer1,weights2)+biases2
    else:
        #首先使用avg_class.average函数来计算得出变量的滑动平均值，然后再计算相应的神经网络前向传播结果。
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

#训练模型
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    #生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    #生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #计算在当前参数下神经网络前向传播的结果，这里是用于计算滑动平均的类为None
    #函数不会使用参数的滑动平均值
    y = inference(x,None,weights1,biases1,weights2,biases2)
    #定义存储训练轮数的变量。这个变量不需要计算滑动平均类。所以改成不可训练类型。
    global_step = tf.Variable(0,trainable=False)

    #给定滑动名平均衰减率和训练轮数，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    #在所有代表神经网络参数的变量上使用滑动平均。
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #计算使用了滑动平均之后的前向传播结果
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

    #计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))

    #计算交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算正则化损失
    regularization = regularizer(weights1)+regularizer(weights2)
    #总损失等于交叉熵和正则化损失的和
    loss = cross_entropy_mean+regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    #使用优化算法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    #一次完成多个操作，通过反向传播来更新神经网络中的参数，又更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #下面的运算首先将bool值转换为实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32));

    #初始化回话并开始训练
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #准备验证数据
        validata_feed = {x: mnist.validation.images,y_: mnist.validation.labels}

        #测试数据
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}

        #迭代神经网络
        starttime = datetime.datetime.now()
        for i in range(TRAINGING_STEPS):
            #每一千轮输出数据
            if(i%1000==0):
                validate_acc = sess.run(accuracy,feed_dict=validata_feed)
                print("After %d traning step(s),validation accuracy""using average model is %g" % (i,validate_acc))

            #产生这一轮使用的batch数据集
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        #训练结束，在测试数据检测最终正确率
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print(TRAINGING_STEPS,test_acc)

        endtime = datetime.datetime.now()
        print("总耗时 %g" % (endtime - starttime).seconds +" 秒")

#主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("D:/mnist_dataset/",one_hot=True)
    train(mnist)

#tensorflow提供主程序入口，如下
if __name__ == '__main__':
    tf.app.run()