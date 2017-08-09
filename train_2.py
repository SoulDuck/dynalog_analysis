import tensorflow as tf
import numpy as np
import os,sys,glob
import matplotlib
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

root_path, names, files = os.walk('./divided_log').next()
dir_paths = map(lambda name: os.path.join(root_path, name), names)
train_xs , train_ys=data.merge_all_data(dir_paths[:10])
test_xs , test_ys=data.merge_all_data(dir_paths[10:11])

train_xs, train_ys, test_xs, test_ys= list(data.get_specified_leaf(leaf_num , train_xs , train_ys , test_xs , test_ys ))
train_xs, train_ys, test_xs, test_ys=data.normalize(train_xs, train_ys, test_xs, test_ys)


if __debug__ == debug_flag_lv1:
    print 'shape train xs', np.shape(train_xs)
    print 'shape test xs', np.shape(test_xs)
    print 'shape train ys', np.shape(train_ys)
    print 'shape test ys', np.shape(test_ys)

n, seq_length , n_col=np.shape(train_xs)



"""
parser=argparse.ArgumentParser()
parser.add_argument('--iter')
parser.add_argument('--learning_rate')
"""

data_dim=3
hidden_dim=10
output_dim=1
learning_rate=0.01
iterations=50000


"""
x_ = tf.placeholder(tf.float32, [None, seq_length , data_dim])
y_ = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)


outputs, _states = tf.nn.dynamic_rnn(cell, x_, dtype=tf.float32)
pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(pred - y_))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    try:
        for i in range(iterations):

            _, step_loss = sess.run([train, loss], feed_dict={
                x_: train_xs, y_: train_ys})
            print("[step: {}] loss: {}".format(i, step_loss))

        # Test step

        test_predict, outputs_ = sess.run([pred, outputs], feed_dict={x_: test_xs})
        rmse_val = sess.run(rmse, feed_dict={targets: test_ys, predictions: test_predict})
        print outputs_, 'outputs shape', np.shape(outputs_)
        print("RMSE: {}".format(rmse_val))
        print test_predict
        raise KeyboardInterrupt
    except KeyboardInterrupt as kbi:
        utils.plot_xy(test_predict=test_predict , test_ys=test_ys)
"""



