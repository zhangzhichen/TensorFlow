import tensorflow as tf
#生成模拟数据集
from numpy.random import RandomState
#定义数据batch的大小
batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
#调用激活函数,损失函数和反向传播算法
y = tf.sigmoid(y)
#tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。下面例子不指定，计算所有维度
#tf.clip_by_value 压缩在1e-10,1.0之间
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))+ (1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
#cross_entropy = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
#此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#随机数生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
#定义规则，所有x1+x2<1都是正样本
#其它为负样本
Y = [[int(x1+x2<1)] for (x1,x2) in X]
#初始化所有常量
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    print(rdm)

    #设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size个样本进行训练。
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)

        #通过选取的样本训练神经网络并更新参数。
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000 == 0 and i/1000>0:
            #每隔一段时间计算在所有数据上的交叉熵并输出。
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross entropy on all data is %g" %(i,total_cross_entropy))
            print(sess.run(w1))
            print(sess.run(w2))