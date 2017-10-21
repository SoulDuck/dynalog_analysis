#-*- coding:utf-8 -*-
import numpy as np
import preprocessing
import sys, os , glob
import time
import utils
import random
import analysis
import matplotlib.pyplot as plt
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


def get_ep_all(dir_paths , leaf_n=30 , seq_length=7):
    for i,path in enumerate(dir_paths):
        ep=get_ep(path , leaf_n=leaf_n , seq_length=seq_length)
        if i ==0:
            tmp_arr=ep[:]
        else:
            tmp_arr=np.hstack((tmp_arr, ep[:] ))
            print tmp_arr
    return tmp_arr

def get_ep(dir_path , leaf_n=30 , seq_length=7):
    f=open(os.path.join(dir_path , 'ep.txt'))
    lines=f.readlines()
    for i, line in enumerate(lines):

        #print line.split(',')[:-1]
        if i ==0:
            values=map(int , line.split(',')[:-1])
        else:
            values=np.vstack((values , map(int , line.split(',')[:-1])))
    print "### value shape ###"
    print np.shape(values)

    if leaf_n ==None:
        return values
    else:
        print np.shape(values[seq_length+1: , leaf_n]) #ep을 기준으로 ap의 상대적인 위치을 기록한다 그래서 +1 을 더한다
        return values[seq_length+1: , leaf_n]




def next_batch(x , y , batch_size):
    indices=np.random.permutation(len(y))
    batch_xs=x[indices[:batch_size]]
    batch_ys = y[indices[:batch_size]]
    return batch_xs , batch_ys



def get_error_indices(ep , ap , leaf_n):
    """
    :param ep:
    :param ap:
    :param leaf_n:
    :return: return ep_larger_indices , ep_less_indices  , ep_same_indices
    """

    f_ep=open(ep , 'r')
    f_ap=open(ap , 'r')
    ep_lines = f_ep.readlines()
    ap_lines = f_ap.readlines()

    ep_larger=[] # ep_larger than ap
    ep_less=[] # ep less than ap
    ep_same=[]
    assert len(ep_lines) == len(ap_lines) ,'{} | {}'.format(len(ep_lines),len(ap_lines))

    ep_lines=map( lambda line : map(int ,line.split(',')[:-1]), ep_lines) # line.split(',')[:-1] for delete \n
    ap_lines = map(lambda line: map(int, line.split(',')[:-1]), ap_lines)  # line.split(',')[:-1] for delete \n
    ep_lines=np.asarray(ep_lines)
    ap_lines = np.asarray(ap_lines)
    ep_=ep_lines[:,leaf_n]
    ap_ = ap_lines[:, leaf_n]
    a=np.zeros(len(ep_))

    #numpy indexing
    ep_larger_indices=np.squeeze(np.where(ep_ > ap_))
    ep_less_indices = np.squeeze(np.where(ep_ < ap_))
    ep_same_indices = np.squeeze(np.where(ep_ == ap_))

    assert len(ep_larger_indices)+len(ep_less_indices)+len(ep_same_indices) == len(ep_)\
        ,'# ep larger indices : {} , # ep less indices : {} , # ep same indices {} , # total ep {}'.format(\
        len(ep_larger_indices) , len(ep_less_indices) , len(ep_same_indices) , len(ep_))
    print len(ep_larger_indices)
    print len(ep_less_indices)
    print len(ep_same_indices)
    return ep_, ap_ , ep_larger_indices , ep_less_indices  , ep_same_indices

def plot_ep_ap_graph(ep, ap ,leaf_n):
    print 'plot_ep_ap_graph'
    ep_ , ap_ ,ep_larger , ep_less , ep_same = get_error_indices(ep ,ap ,leaf_n)

    ep_large_diff=ep_[ep_larger] - ap_[ep_larger]
    print ep_large_diff
    print len(ep_large_diff)
    print ep_large_diff.max()

    ep_less_diff = ep_[ep_less] - ap_[ep_less]
    print ep_less_diff
    print len(ep_less_diff)
    print ep_less_diff.min()

    plt.figure(figsize=(50, 10))
    plt.scatter(x = ep_larger , y=ep_[ep_larger] , color='red' , label='ep larger than ap',)
    plt.scatter(x = ep_less , y=ep_[ep_less] ,color='blue' ,label='ep less than ap')
    plt.scatter(x = ep_same , y=ep_[ep_same],color='green' ,label = 'ep same as ap ')
    plt.show()
    plt.savefig('./ep_diff_from_ap.png')
    plot_ep_ap_graph(ep=ep_ , ap=ap_ , leaf_n=30)
    plt.plot(range(len(ap_)) , ap_)
    plt.savefig('./ap_.png')

def get_min_max(*datum):
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
    return min_ ,max_

def normalize(*datum):
    debug_flag_lv0 = True
    if __debug__ == debug_flag_lv0:
        print 'start :### debug | data.py | normalize'
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

    if __debug__ == debug_flag_lv0:

        print 'min value', min_
        print 'max value', max_
        print 'normalized data sample : ' , data[:1]
        print 'end :### debug | data.py | normalize'
    return ret_list


def get_data(folder_path):
    debug_flag_lv0 = False
    if __debug__ == debug_flag_lv0:
        print '### debug | data.py | get_data'
    x_data = np.load(os.path.join(folder_path ,'x_data.npy'))
    y_data = np.load(os.path.join(folder_path ,'y_data.npy'))
    return x_data, y_data

def merge_all_data(dir_paths):
    debug_flag_lv0=True
    if __debug__ == debug_flag_lv0:
        print 'start : ### debug | data.py | merge_all_data'
    print 'the # of input paths:',len(dir_paths)
    xs=None;ys=None;
    for i,dir_path in enumerate(dir_paths):
        x,y=get_data(dir_path)
        if i==0:
            xs=x
            ys=y
        else:
            xs=np.concatenate([xs, x], axis=0)
            ys=np.concatenate([ys, y], axis=0)
    if __debug__ == debug_flag_lv0:
        print 'merged x shape :',np.shape(xs)
        print 'merged y shape :',np.shape(ys)
        print 'end : ### debug | data.py | merge_all_data'
    return xs,ys

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
    # /divided_log 에 있는 x_data , y_data 을 다 긁어 모은다. 붙인다(concatenate)
    # 그리고 반혼한다
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
        ret_data=data[:, leaf_num-1] # width 가 3임으로 leaf_num이 30 이라면 31번째 리프를 보고 싶은것일텐데 -1을 안하면 32번째 리프가 불려오게 된다
        if len(np.shape(data))==2:
            ret_data=ret_data.reshape([-1,1])
        ret_list.append(ret_data)
    if __debug__ == debug_flag_lv0:
        print 'end : ###end debug | data.py | get_specified_leaf'
    return ret_list
#def next_batch(x,y,batch_size):


if __name__ == '__main__':
    """
    #merge_xy_data(limit=2)
    root_path , names , files=os.walk('./divided_log').next()
    dir_paths=map(lambda name : os.path.join(root_path , name) , names )
    #xs,ys=merge_all_data(dir_paths[:2])
    dir_paths=dir_paths[:1]
    ap_paths = os.path.join(dir_paths[0], 'ap.txt')
    ep_paths = os.path.join(dir_paths[0], 'ep.txt')
    ep_, ap_, ep_larger_indices, ep_less_indices, ep_same_indices=get_error_indices(ep=ep_paths , ap=ap_paths , leaf_n=30)

    plt.close()
    plt.figure(figsize=(10, 50))
    plt.scatter(x=ep_larger_indices ,y= ep_[ep_larger_indices]  ,s=2)
    plt.scatter(x=ep_larger_indices ,y= ap_[ep_larger_indices] ,s=2)
    plt.savefig('tmp3.png')
    """
    get_ep('./divided_log/A20170614151153_RT02526/ep.txt')

