import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os,sys,glob
import data


debug_flag_lv0=False
debug_flag_lv1=True
debug_flag_lv2=False
if __debug__ == debug_flag_lv0:
    print '###debug | train.py | '


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



_,f_names,files=os.walk('./divided_log').next()

for i,f_name in enumerate(f_names[:2]):
    x_data=np.load('divided_log/'+f_name+'/x_data.npy')
    y_data=np.load('divided_log/'+f_name+'/y_data.npy')
    train_x , test_x , train_y , test_y =data.get_train_test_xy_data(x_data, y_data , 0.1)
    if i==0:
        train_xs=train_x
        test_xs =test_x
        train_ys=train_y
        test_ys =test_y
    else:
        train_xs=np.concatenate([train_xs , train_x] , axis=0)
        test_xs=np.concatenate([test_xs , test_x] , axis=0)
        train_ys=np.concatenate([train_ys, train_y], axis=0)
        test_ys=np.concatenate([test_ys, test_y], axis=0)
assert len(train_xs)==len(train_ys)
assert len(test_ys)==len(test_xs)

# 3rd leaf
leaf_num=30
train_x=train_xs[:,leaf_num]
test_x=test_xs[:,leaf_num]
train_y=train_ys[:,leaf_num].reshape([-1,1])
test_y=test_ys[:,leaf_num].reshape([-1,1])




if __debug__ == debug_flag_lv1:
    print 'sample ',leaf_num, train_xs[:10]
    print 'train_xs shape', np.shape(train_xs)
    print 'train_xs shape', np.shape(test_xs)
    print 'train_ys shape', np.shape(train_ys)
    print 'train_ys shape', np.shape(test_ys)

"""
n, seq_length , n_col=np.shape(train_x)
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
"""
