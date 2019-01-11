import tensorflow as tf
#定义变量相加的计算
v1 = tf.Variable(tf.constant(1.0,shape=[1],name='v1'))
v2 = tf.Variable(tf.constant(4.0,shape=[1],name='v2'))
result1 = v1+v2

saver = tf.train.Saver()
#保存为json格式
saver.export_meta_graph("D:\saver\chapter5/model.ckpt.meda.json",as_text=True)