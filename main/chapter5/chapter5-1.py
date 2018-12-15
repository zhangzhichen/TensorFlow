from tensorflow.examples.tutorials.mnist import input_data
#载入数据集。
mnist = input_data.read_data_sets("D:/mnist_dataset/",one_hot=True)

#打印size
print("Trabning data size:",mnist.train.num_examples)

#打印Validating data size
print("Validating data size:",mnist.validation.num_examples)

#打印测试数据
print("Testing data size:",mnist.test.num_examples)

#打印example traning data
print("Example training data:",mnist.train.images[0])

#随机选取数据集,随机梯度下降算法
batch_size = 100
xs,ys = mnist.train.next_batch(batch_size)

print("X shape:",xs.shape)
print("Y shape:",ys.shape)