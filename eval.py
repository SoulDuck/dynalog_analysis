import tensorflow as tf
import numpy as np




def eval(x , y , sess ,saved_model_folder):
    sess =tf.Session()
    saver=tf.train.import_meta_graph(meta_graph_or_file=meta_graph_or_file)
    saver.restore(sess = sess  , save_path=save_path)
    tf.get_default_graph()


    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    cam_ = tf.get_default_graph().get_tensor_by_name('classmap_reshape:0')
    top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
    phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    y_conv = tf.get_default_graph().get_tensor_by_name('y_conv:0')