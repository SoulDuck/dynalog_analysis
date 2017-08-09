import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os,sys,glob
import data
import argparse
debug_flag_lv0=False
debug_flag_lv1=True
debug_flag_lv2=False




"""
--- 3 --
 _  _  _ 
| || || | |
| || || | 7
|_||_||_| |
 c1 c2 c3

1.left leaf
2.target leaf
3.right leaf
"""

if __debug__ == debug_flag_lv0:
    print '###debug | train.py | '

train_xs,train_ys,test_xs,test_ys=data.merge_xy_data()
data.merge_xy_data()

if __debug__ == debug_flag_lv1:
    print 'train_xs shape', np.shape(train_xs)
    print 'train_xs shape', np.shape(test_xs)
    print 'train_ys shape', np.shape(train_ys)
    print 'train_ys shape', np.shape(test_ys)


n, seq_length , n_col=np.shape(train_xs)
data_dim=3
hidden_dim=10
output_dim=1
learning_rate=0.01
iterations=50000



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
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
            x_: train_x, y_: train_y})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step

    test_predict, outputs_ = sess.run([pred, outputs], feed_dict={x_: test_x})
    rmse_val = sess.run(rmse, feed_dict={targets: test_y, predictions: test_predict})
    print outputs_, 'outputs shape', np.shape(outputs_)
    print("RMSE: {}".format(rmse_val))
    print test_predict

    # plot predictions

    plt.plot(test_y)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("leaf control point")
    plt.show()

