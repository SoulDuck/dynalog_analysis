#-*- coding:utf-8 -*-
import tensorflow as tf
import analysis
import os
import data
import numpy  as np


# add ep 와 ap의 그래프를 그려주는 plot 함수를 추가해야 한다
def eval(x , y , error_range_percent , model_path , normalize_factor, model_extension=None):

    sess =tf.Session()
    saver=tf.train.import_meta_graph(meta_graph_or_file=model_path+'.meta')
    if model_extension ==None:
        model_extension=''
    saver.restore(sess = sess  , save_path=model_path+model_extension)
    tf.get_default_graph()


    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    pred_op= tf.get_default_graph().get_tensor_by_name('pred:0')
    loss_op= tf.get_default_graph().get_tensor_by_name('loss:0')
    fetches=[pred_op , loss_op]
    pred , loss=sess.run(fetches=fetches , feed_dict={x_:x , y_:y})

    #print y[:10]
    #print pred[:10]
    pred=pred*normalize_factor
    x=x*normalize_factor
    y=y*normalize_factor
    print np.shape(x)
    print np.shape(y)
    print np.max(pred)
    print np.max(y)
    print np.max(pred)


    acc=analysis.get_acc_with_ep(ep=x ,true=y , pred=pred , error_range_percent=error_range_percent)
    sess.close()

    return pred , loss , acc


if '__main__'==__name__:
    normalize_factor=10000.
    TEST_SET = ['./divided_log/A20170615085606_RT02473', './divided_log/A20170615083340_RT02494',
                './divided_log/A20170620103113_RT02468']
    test_xs, test_ys = data.merge_all_data(TEST_SET)
    test_xs=test_xs[:,30,:,:]
    test_ys = test_ys[:, 30].reshape([-1,1])
    test_xs=test_xs/normalize_factor
    test_ys=test_ys/normalize_factor

    pred , loss , acc =eval(test_xs, test_ys,4,  model_path='models/1/40200', normalize_factor= normalize_factor)
    print np.shape(pred)
    f=open('pred.txt' ,'w')
    for i in range(len(pred)):
        f.write(str(pred[i][0])+'\n')
    f.close()

    test_ys=test_ys*normalize_factor
    f = open('ap.txt', 'w')
    for i in range(len(test_ys)):
        f.write(str(test_ys[i][0]) + '\n')
    f.close()








