import numpy as np
import preprocessing
import sys, os , glob
import time
import utils
import random
def merge_xy_data(folder_path='./divided_log/'):
    start_time=time.time()
    debug_flag=True
    debug_flag_lv1=True
    debug_flag_lv2=True


    if __debug__ == debug_flag:
        print '### debug | data.py | merge_xy_data'
    merged_x_data=[]
    merged_y_data=[]
    path , subfolders , files = os.walk(folder_path).next()
    for i,subfolder in enumerate(subfolders[:]):
        utils.show_processing(i,len(subfolders))
        target_path=os.path.join(folder_path , subfolder)
        x_data=np.load(os.path.join(target_path , 'x_data.npy'))
        y_data = np.load(os.path.join(target_path, 'y_data.npy'))

        if i==0:
            merged_x_data = x_data
            merged_y_data = y_data
        else:
            merged_x_data=np.concatenate([merged_x_data ,x_data] ,axis=0)
            merged_y_data=np.concatenate([merged_y_data ,y_data]  ,axis=0)
    if __debug__ == debug_flag_lv1:
        print 'merged x data shape ',np.shape(merged_x_data)
        print 'merged y data shape ',np.shape(merged_y_data)
        print 'merged time ' , time.time()-start_time

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


if __name__ == '__main__':
    merge_xy_data()
