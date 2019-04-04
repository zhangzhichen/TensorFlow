import tensorflow as tf
import matplotlib.pyplot as plt

#读取图像的原始数据
image_raw_data = tf.gfile.GFile("D:\picture/tiger.jpeg","rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    #打印图像矩阵
    print(img_data.eval())

    #使用可视化工具
    plt.imshow(img_data.eval())
    plt.show()

    #将数据类型转换为实数
    #img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)

    #将数据重新保存，得到和原图形一样的图像
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("D:\output//tiger.jpg","wb") as f:
        f.write(encoded_image.eval())
