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
debug_flag_lv0=False
debug_flag_lv1=True
debug_flag_lv2=False

if __debug__ == debug_flag_lv0:
    print '###debug | train.py | '


leaf_num=30
n_train=-3
batch_size=60
root_path, names, files = os.walk('./divided_log').next()
dir_paths = map(lambda name: os.path.join(root_path, name), names)
print 'dir paths : ',dir_paths[:]
print 'length',len(dir_paths)
dir_paths=dir_paths[:5]

dir_paths=dir_paths[:]
train_xs , train_ys=data.merge_all_data(dir_paths[:n_train])
test_xs , test_ys=data.merge_all_data(dir_paths[n_train:])
print dir_paths[n_train:]
train_xs, train_ys, test_xs, test_ys= list(data.get_specified_leaf(leaf_num , train_xs , train_ys , test_xs , test_ys ))

min_ , max_ =data.get_min_max(train_xs, train_ys, test_xs, test_ys)
print 'min', min_ , 'max' ,max_
normalize_factor=10000.
train_xs=train_xs/normalize_factor
test_xs=test_xs/normalize_factor
train_ys=train_ys/normalize_factor
test_ys=test_ys/normalize_factor
#
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
init_lr=0.1
reduced_lr1=30000
reduced_lr2=80000
reduced_lr3=130000
iterations=100
check_point=100
n_cell=3



x_ = tf.placeholder(tf.float32, [None, seq_length , data_dim])
y_ = tf.placeholder(tf.float32, [None, 1])
lr_=tf.placeholder(tf.float32, name='learning_rate')
# build a LSTM network

cell = lstm(hidden_dim=hidden_dim)
multi_cell=tf.contrib.rnn.MultiRNNCell([lstm(hidden_dim)  for _ in range(n_cell)])
outputs, _states = tf.nn.dynamic_rnn(multi_cell, x_, dtype=tf.float32)
print 'Cell shape : ',outputs
pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output
print 'FC layer output shape :',pred
# cost/loss
loss = tf.reduce_sum(tf.square(pred - y_))  # sum of the squares
tf.summary.scalar('accuracy', loss)
tf.summary.scalar('learning_rate', lr_)
# optimizer
optimizer = tf.train.AdamOptimizer(lr_)
train = optimizer.minimize(loss)

targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

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
    try:
        for i in range(iterations):

            if i<= reduced_lr1:
                learning_rate=init_lr
            elif i <= reduced_lr2:
                learning_rate = 0.001
            elif i <= reduced_lr3:
                learning_rate = 0.0001
            else:
                learning_rate=0.000001

            if i%check_point ==0:
                #batch_xs , batch_ys = data.next_batch(train_xs , train_ys , batch_size)
                #print np.shape(batch_xs) , np.shape(batch_ys)
                test_predict, outputs_, test_loss , merged_summaries= sess.run([pred, outputs, loss ,merged ], feed_dict={x_: test_xs , y_ : test_ys , lr_:learning_rate})
                print("[step: {}] test loss: {}".format(i, test_loss))
                print("[step: {}] train loss: {}".format(i, train_loss))
                test_writer.add_summary(merged_summaries , i)
                utils.plot_xy(test_predict=test_predict, test_ys=test_ys , savename='./graph/dynalog_result_'+str(i)+'.png')
                saver.save(sess=sess , save_path='./models/model' , global_step=i)
            _, train_loss , merged_summaries = sess.run([train, loss , merged], feed_dict={x_: train_xs, y_: train_ys, lr_:learning_rate})

            train_writer.add_summary(merged_summaries, i)
        # Test step
        test_predict, outputs_  , test_loss = sess.run([pred, outputs,loss], feed_dict={x_: test_xs,y_ : test_ys})

        rmse_val = sess.run(rmse, feed_dict={targets: test_ys, predictions: test_predict})
        print outputs_, 'outputs shape', np.shape(outputs_)
        print("RMSE: {}".format(rmse_val))
        print test_predict
        raise KeyboardInterrupt
    except KeyboardInterrupt as kbi:
        pred=test_predict*normalize_factor
        test_ys=test_ys*normalize_factor

        analysis.analysis_result(true = test_ys , pred = pred , error_range_percent=5)
        test_ys = test_ys * (max_ - min_) + min_
        utils.plot_xy(test_predict=test_predict, test_ys=test_ys, savename='./dynalog_result_last' + '.png')

        #np.save('./test_ep.npy',test_xs[])
