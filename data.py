import numpy as np
import preprocessing
import sys, os , glob
import time
import utils
import random

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
def normalize(*datum):
    ret_list = []
    for i,data in enumerate(datum):
        assert type(data).__module__==np.__name__
        if i==0:
            min_=np.min(data)
            max_=np.max(data)
        else:
            if np.min(data) < min_:
                min_=np.min(data)
            if np.max(data) > max_:
                max_ = np.max(data)

    for i, data in enumerate(datum):
        data=(data-min_) / (max_ - min_)
        ret_list.append(data)

    print 'min value', min_
    print 'max value', max_
    print data[:1]

    return ret_list




def get_train_test_xy_data(x_data , y_data , test_ratio):
    start_time=time.time()
    debug_flag=True
    debug_flag_lv1=True
    debug_flag_lv2=True
    if __debug__ == debug_flag:
        print '### debug | data.py | get_train_test_xy_data'

    assert len(x_data) == len(y_data)
    n_test=int(len(x_data)*test_ratio *test_ratio)

    indices=random.sample(range(len(x_data)) , len(x_data))

    test_x=x_data[indices[:n_test]]
    train_x=x_data[indices[n_test:]]
    test_y = y_data[indices[:n_test]]
    train_y = y_data[indices[n_test:]]

    assert len(test_x) == len(test_y)
    assert len(train_x) == len(train_y)

    if __debug__ == debug_flag_lv1:
        print '# test',n_test
        print 'shape train x', np.shape(train_x)
        print 'shape test x',np.shape(test_x)
        print 'shape train y', np.shape(train_y)
        print 'shape test y',np.shape(test_y)
    return train_x, test_x , train_y , test_y



def merge_xy_data(root_dir= './divided_log' , limit=None):
    debug_flag_lv0 = True
    if __debug__ == debug_flag_lv0:
        print 'start : ###debug | data.py | get_specified_leaf'
        print 'limit',limit
    _,f_names,files=os.walk(root_dir).next()
    for i,f_name in enumerate(f_names[:limit]):
        x_data=np.load('divided_log/'+f_name+'/x_data.npy')
        y_data=np.load('divided_log/'+f_name+'/y_data.npy')
        train_x , test_x , train_y , test_y =get_train_test_xy_data( x_data, y_data , 0.1)
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
    if __debug__ == debug_flag_lv0:
        print 'merged train_xs shape:',np.shape(train_xs)
        print 'merged test_xs shape:' , np.shape(test_xs)
        print 'merged train_ys shape:', np.shape(train_ys)
        print 'merged test_ys shape:' , np.shape(test_ys)
        print 'end : ###debug | data.py | get_specified_leaf'
    return train_xs, train_ys, test_xs, test_ys
"""
    # 3rd leaf
    train_x=train_xs[:,leaf_num]
    test_x=test_xs[:,leaf_num]
    train_y=train_ys[:,leaf_num].reshape([-1,1])
    test_y=test_ys[:,leaf_num].reshape([-1,1])
"""

def get_specified_leaf(leaf_num , *datum):
    ret_list=[]
    debug_flag_lv0=True
    if __debug__ == debug_flag_lv0:
        print 'start : ###debug | data.py | get_specified_leaf'
    for data in datum:
        assert type(data).__module__ == np.__name__ #check input data ,is numpy data or not

        ret_data=data[:, leaf_num]
        if len(np.shape(data))==2:
            ret_data=ret_data.reshape([-1,1])
        ret_list.append(ret_data)

    if __debug__ == debug_flag_lv0:
        print 'end : ###end debug | data.py | get_specified_leaf'
    return ret_list

#def next_batch(x,y,batch_size):

if __name__ == '__main__':
    merge_xy_data(limit=2)
