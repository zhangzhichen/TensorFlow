import tensorflow as tf
#作用域
with tf.variable_scope("root"):
    #获取reuse属性
    print(tf.get_variable_scope().reuse)
    #嵌套上下文管理器
    with tf.variable_scope("foo",reuse=True):
        #此时上下文为指定的True
        print(tf.get_variable_scope().reuse)

        #新建一个取值，但不指定resue，与外层保持一致
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)

    #外层还是false
    print(tf.get_variable_scope().reuse)