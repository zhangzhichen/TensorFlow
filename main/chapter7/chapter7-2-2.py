import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#给定一张图像，随机调整图像的色彩，因为调整亮度，对比度，饱和度和色相的顺序会影响最后得到的结果
#所以可以定义多种不同的顺序。
def distort_color(image,color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image,max_delta=32./255.)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image,max_delta=32./255.)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)

    return tf.clip_by_value(image,0.0,1.0)

def preprocess_for_train(image,new_shape,bbox):
    #如果没有标注框，则认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])

    #转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)

    #随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distorted_image = tf.slice(image,bbox_begin,bbox_size)

    #将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的
    distorted_image = tf.image.resize_images(distorted_image,new_shape,method = np.random.randint(4))

    #随机左右翻转
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    #使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image,np.random.randint(2))

    return distorted_image

image_raw_data = tf.gfile.GFile("D:\picture/cat.jpg","rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    #循环6次获取6种不同的图像
    for i in range(6):
        #将图像尺寸调整为299*299
        result = preprocess_for_train(img_data,[299,299],boxes)
        plt.imshow(result.eval())
        plt.show()
