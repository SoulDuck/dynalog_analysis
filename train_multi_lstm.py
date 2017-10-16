#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os,sys,glob
import matplotlib
import analysis
if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data
import argparse
import utils
import eval
debug_flag_lv0=False
debug_flag_lv1=True
debug_flag_lv2=False
debug_flag_test=True
if __debug__ == debug_flag_lv0:
    print '###debug | train.py |'

leaf_num=30
n_train=-3
batch_size=60
root_path, names, files = os.walk('./divided_log').next()
dir_paths = map(lambda name: os.path.join(root_path, name), names)
print 'dir paths : ',dir_paths[:]
print 'length',len(dir_paths)

if debug_flag_test:
    dir_paths=dir_paths[:5]
else:
    dir_paths = dir_paths[:]

train_xs , train_ys=data.merge_all_data(dir_paths[:n_train])
test_xs , test_ys=data.merge_all_data(dir_paths[n_train:])

print '##### directory paths #####'
print dir_paths[n_train:]
train_xs, train_ys, test_xs, test_ys= list(data.get_specified_leaf(leaf_num , train_xs , train_ys , test_xs , test_ys))
ep=data.get_ep_all(dir_paths[n_train:] , leaf_n=leaf_num)

""" ep 하고 ap 는 데이터를 만들때 7개의 row 을 힉습시키고 그 다음 위치를 예측하는 형태로 x_data , y_data 을 만들었다
하지만 위 ep는 ap와 ep을 비교해 graph을 그리는 데 사용할 것이기 때문에 기존의 x_Data에서 불러온 ep데이터와 달리 
ap와 같은 데이터 위치를 가지고있다 .
 
default 셋팅으로 길이는 7 너비는 3 으로 설정하였기 때문에 이 ep 데이터는 8번째 행부터 시작하며 데이터 지정한 열의 데이터를 가지고 온다.
데이터를 일일이 눈으로 보면서 검증했다
"""

assert len(ep) == len(test_ys) , len(test_ys)
print 'ep',ep[:8]
print 'test_ys',test_ys[:8]
print 'test_xs',test_xs[:8]

min_ , max_ =data.get_min_max(train_xs, train_ys, test_xs, test_ys)
print 'min', min_ , 'max' ,max_

normalize_factor=10000.
train_xs=train_xs/normalize_factor
test_xs=test_xs/normalize_factor
train_ys=train_ys/normalize_factor
test_ys=test_ys/normalize_factor

print train_xs.max()
print train_xs.min()
print test_xs.max()
print test_xs.min()
# train_xs, train_ys, test_xs, test_ys=data.normalize(train_xs, train_ys, test_xs, test_ys)
if __debug__ == debug_flag_lv1:
    print 'shape train xs', np.shape(train_xs)
    print 'shape test xs', np.shape(test_xs)
    print 'shape train ys', np.shape(train_ys)
    print 'shape test ys', np.shape(test_ys)
n, seq_length , n_col=np.shape(train_xs)

def lstm(hidden_dim):
    return tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)


"""
parser=argparse.ArgumentParser()
parser.add_argument('--iter')
parser.add_argument('--learning_rate')
"""

data_dim=3
hidden_dim=30
output_dim=1
init_lr=0.01
reduced_lr1=20000
reduced_lr2=50000
reduced_lr3=80000
if debug_flag_test:
    iterations=101
else:
    iterations=150000
check_point=100
n_cell=3



x_ = tf.placeholder(tf.float32, [None, seq_length , data_dim] , name='x_')
y_ = tf.placeholder(tf.float32, [None, 1] , name ='y_')
lr_=tf.placeholder(tf.float32, name='learning_rate')
# build a LSTM network

cell = lstm(hidden_dim=hidden_dim)
multi_cell=tf.contrib.rnn.MultiRNNCell([lstm(hidden_dim)  for _ in range(n_cell)])
outputs, _states = tf.nn.dynamic_rnn(multi_cell, x_, dtype=tf.float32)
print 'Cell shape : ',outputs
pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output
pred=tf.identity(pred , name='pred')
print 'FC layer output shape :',pred
# cost/loss
loss = tf.reduce_sum(tf.square(pred - y_) , name='loss')  # sum of the squares
tf.summary.scalar('accuracy', loss)
tf.summary.scalar('learning_rate', lr_)
# optimizer
optimizer = tf.train.AdamOptimizer(lr_)
train = optimizer.minimize(loss , name='train_op')

if not os.path.isdir('./graph'):
    os.mkdir('./graph')
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(logdir='./logs/train')
    test_writer = tf.summary.FileWriter(logdir='./logs/test')
    init = tf.global_variables_initializer()
    sess.run(init)
    # Training step
    train_loss=0
    best_acc=0
    best_loss=0
    tmp_loss=0
    try:

        for i in range(iterations):
            print i
            if i<= reduced_lr1:
                learning_rate=init_lr
            elif i <= reduced_lr2:
                learning_rate = 0.001
            elif i <= reduced_lr3:
                learning_rate = 0.0001
            else:
                learning_rate=0.00001
            if i%check_point ==0:
                #batch_xs , batch_ys = data.next_batch(train_xs , train_ys , batch_size)
                #print np.shape(batch_xs) , np.shape(batch_ys)
                test_predict, outputs_, test_loss , merged_summaries= sess.run([pred, outputs, loss ,merged ], feed_dict={x_: test_xs , y_ : test_ys , lr_:learning_rate})
                print("[step: {}] test loss: {}".format(i, test_loss))
                print("[step: {}] train loss: {}".format(i, train_loss))
                test_writer.add_summary(merged_summaries , i)
                utils.plot_xy(test_predict=test_predict, test_ys=test_ys , savename='./graph/dynalog_result_'+str(i)+'.png')
                acc=analysis.get_acc_with_ep(ep= test_xs , true = test_ys*normalize_factor , pred = test_predict*normalize_factor , error_range_percent=5)
                print("[step: {}] test acc: {}".format(i, acc))
                summary=tf.Summary(value = [tf.Summary.Value(tag='accuracy %s'%'test' , simple_value =float(acc))])
                train_writer.add_summary(summary=summary , global_step=i )
                if best_acc < acc:
                    best_acc = acc
                    tmp_loss = test_loss
                    saver.save(sess=sess, save_path='./models/acc_{}_loss_{}'.format(str(best_acc)[:4] , str(tmp_loss)[:4]), global_step=i)
                    print 'model saved'
                elif best_acc == acc:
                    if best_loss > test_loss:
                        best_loss = test_loss
                        saver.save(sess=sess, save_path='./models/acc_{}_loss_{}.ckpt'.format(str(best_acc)[:4], str(best_loss)[:4]),
                                   global_step=i)
                        print 'model saved'
            _, train_loss , merged_summaries = sess.run([train, loss , merged], feed_dict={x_: train_xs, y_: train_ys, lr_:learning_rate})
            train_writer.add_summary(merged_summaries, i)
        # Test step
        test_predict, outputs_  , test_loss = sess.run([pred, outputs,loss], feed_dict={x_: test_xs,y_ : test_ys})
        loss_val = sess.run(loss, feed_dict={y_: test_ys, pred: test_predict})
        print outputs_, 'outputs shape', np.shape(outputs_)
        print("RMSE: {}".format(loss_val))
        #print test_predict
        raise KeyboardInterrupt
    except KeyboardInterrupt as kbi:
        print '#### result ###'
        pred=test_predict*normalize_factor
        test_ys=test_ys*normalize_factor
        analysis.analysis_result(true = test_ys , pred = pred , error_range_percent=5)
        test_ys = test_ys * (max_ - min_) + min_
        utils.plot_xy(test_predict=test_predict, test_ys=test_ys, savename='./dynalog_result_last' + '.png')
        print '########'
        print test_ys[:10]
        print test_predict[:10]
        sess.close()
        print 'start evaluation'
        eval.eval(x=test_xs  , y=test_ys , error_range_percent=5 , model_path='./models/acc_{}_loss_{}-100'.format(str(best_acc)[:4] , str(tmp_loss)[:4]))

        #np.save('./test_ep.npy',test_xs[])
