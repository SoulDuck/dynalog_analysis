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

leaf_num=57
for f in [1,10,20,30,40,50,57]:
    utils.make_dir(str(f))
    n_train=15
    root_path, names, files = os.walk('./divided_log').next()
    dir_paths = map(lambda name: os.path.join(root_path, name), names)

    train_xs , train_ys=data.merge_all_data(dir_paths[:n_train])
    test_xs , test_ys=data.merge_all_data(dir_paths[n_train:])
    print dir_paths[n_train:]
    train_xs, train_ys, test_xs, test_ys= list(data.get_specified_leaf(f , train_xs , train_ys , test_xs , test_ys ))
    min_ , max_ =data.get_min_max(train_xs, train_ys, test_xs, test_ys)
    print 'min', min_ , 'max' ,max_
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
    iterations=10000
    check_point=100



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

        f_train_loss=open('./'+str(f)+'/train_loss.txt' , 'w')
        f_test_loss=open('./'+str(f)+'/test_loss.txt', 'w')



        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        try:
            for i in range(iterations):
                if i%check_point ==0:
                    test_predict, outputs_, test_loss = sess.run([pred, outputs, loss], feed_dict={x_: test_xs , y_ : test_ys})
                    np.save('./' + str(f) + '/'+str(i)+'_predict_normalize.npy', test_predict)
                    np.save('./' + str(f) + '/'+str(i)+'_true_normalizae.npy', test_ys)
                    print("[step: {}] test loss: {}".format(i, test_loss))
                    utils.plot_xy(test_predict=test_predict, test_ys=test_ys , savename='./'+str(f)+'/dynalog_result_'+str(i)+'.png')
                    msg='{} : {}\n'.format(i,test_loss)
                    f_test_loss.write(msg)
                _, train_loss = sess.run([train, loss], feed_dict={x_: train_xs, y_: train_ys})
                msg = '{} : {}\n'.format(i, train_loss)
                f_train_loss.write(msg)
                f_train_loss.flush()
                f_test_loss.flush()

                print("[step: {}] train loss: {}".format(i, train_loss))

            # Test step

            test_predict, outputs_  , test_loss = sess.run([pred, outputs,loss], feed_dict={x_: test_xs,y_ : test_ys})
            rmse_val = sess.run(rmse, feed_dict={targets: test_ys, predictions: test_predict})
            print outputs_, 'outputs shape', np.shape(outputs_)
            print("RMSE: {}".format(rmse_val))
            print test_predict
            raise KeyboardInterrupt
        except KeyboardInterrupt as kbi:
            test_predict=test_predict*(max_ - min_)+min_
            test_ys = test_ys * (max_ - min_) + min_
            np.save('./' + str(f) + '/last_predict.npy' , test_predict)
            np.save('./' + str(f) + '/true_original.npy' , test_ys)
            utils.plot_xy(test_predict=test_predict, test_ys=test_ys,savename='./' + str(f) + '/dynalog_result_last.png')
            #np.save('./test_ep.npy',test_xs[])

    tf.reset_default_graph()


