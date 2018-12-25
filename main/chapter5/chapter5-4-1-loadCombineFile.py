import tensorflow as tf
from tensorflow.python.framework import graph_util

with tf.Session() as sess:
    with tf.gfile.FastGFile("D:\saver\chapter5\combine.pb","rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        result = tf.import_graph_def(graph_def,return_elements=["add:0"])
        print(sess.run(result))