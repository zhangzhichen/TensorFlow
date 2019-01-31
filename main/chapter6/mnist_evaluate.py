import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import main.chapter6.mnist_inference as mnist_inference
import main.chapter6.mnist_train as mnist_train
EVAL_INTERVAL_SECS = 10
BATCH_SIZE = 100
import time
import numpy as np
def evaluate(mnist):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None,
                                              mnist_inference.IMAGE_SIZE,
                                              mnist_inference.IMAGE_SIZE,
                                              mnist_inference.NUM_CHANNEL], name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[None,mnist_inference.OUTPUT_NODE], name='y-input')

        xs, ys = mnist.validation.images, mnist.validation.labels
        reshape_xs = np.reshape(xs, (-1, mnist_inference.IMAGE_SIZE,
                                     mnist_inference.IMAGE_SIZE,
                                     mnist_inference.NUM_CHANNEL))
        print(mnist.validation.labels[0])
        val_feed = {x: reshape_xs, y_: mnist.validation.labels}
        y = mnist_inference.inference(x,False,None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)

        val_to_restore = variable_average.variables_to_restore()

        saver = tf.train.Saver(val_to_restore)
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=val_feed)
                    print('After %s train ,the accuracy is %g'%(global_step,accuracy_score))
                else:
                    print('No Checkpoint file find')
                    # continue
            time.sleep(EVAL_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets('D:\mnist_dataset',one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()