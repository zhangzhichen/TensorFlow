import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("D:\mnist_dataset/", one_hot=True)
tf.reset_default_graph()

# #################################################
im=mnist.test.images[0].reshape((28,28))
#Image和Ndarray互相转换
img= Image.fromarray(im*255)
#jpg可以是RGB模式，也可以是CMYK模式
img= img.convert('RGB')
#保存
img.save(r'D:\output\02.jpg')
plt.imshow(im,cmap="gray")
plt.show()