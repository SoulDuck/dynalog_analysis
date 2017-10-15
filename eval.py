import tensorflow as tf
import numpy as np
import analysis
import data




def eval(x , y , error_range_percent, sess , model_folder , model_name):
    sess =tf.Session()
    model_path=os.path.join(model_folder , model_name)
    saver=tf.train.import_meta_graph(meta_graph_or_file=model_path+'.meta')
    saver.restore(sess = sess  , save_path=model_path+'.ckpt')
    tf.get_default_graph()


    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    pred = tf.get_default_graph().get_tensor_by_name('pred:0')
    loss_op= tf.get_default_graph().get_tensor_by_name('loss:0')
    fetches=[pred , loss_op]
    pred_ , loss_=sess.run(fetches=fetches , feed_dict={x_:x , y_:y})
    acc=analysis.get_acc(true=y , pred=pred_ , error_range_percent=5)

    return acc
