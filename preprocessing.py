import numpy as np
import os , sys , glob

#ep-ap error



def get_trainingData(folder_path , seq_length=7 , save_is=True):
    debug_flag=True
    if __debug__ == debug_flag:
        print '### debug | preprocessing.py | get_trainingData ###'
        print 'hidden param state: seq_length=',seq_length,'save_is=',save_is

    #type(ep).__moduel__ == __name__.np

    leafs = np.load(os.path.join(folder_path,'leaf.npy'))
    n_length , n_leaf , n_param = np.shape(leafs)

    ep_ = leafs[:,:,0]
    ap_ = leafs[:,:,1]

    x_data=[];y_data=[]
    x_data=np.zeros([n_length-seq_length-1 , n_leaf-3+1 , seq_length , 3])
    y_data=np.zeros([n_length-seq_length-1 , n_leaf-3+1])
    for r in range(n_length-seq_length-1):
        x=ep_[r:r+seq_length] # (7,60)
        y=ap_[r+seq_length+1] # (7,60)
        for c in range(n_leaf-3+1):
            x_data[r, c,:,:]=x[:,c:c+3]
            y_data[r,c]=y[c+1]

    np.save(os.path.join(folder_path ,'x_data.npy'),x_data)
    np.save(os.path.join(folder_path, 'y_data.npy'), x_data)
    assert len(x_data) == len(y_data)
    if __debug__ == debug_flag:
        print 'x_data , and y_data was saved'
    if __debug__ == debug_flag:
        print 'x  data length :,',np.shape(x_data)
        print 'y  data length :,',np.shape(y_data)
        print 'leaf shape :',np.shape(leafs)
        print 'ap shape :', np.shape(ep_)
        print 'ep shape :', np.shape(ap_)
        print '###check###'
        print y_data[0]
        print x_data[0,11]
    return x_data , y_data


    #type(ep).__moduel__ == __name__.np



if __name__ =='__main__':
    root_dir='./divided_log'
    path, subfolders , files=os.walk(root_dir).next()
    for subfolder in subfolders:
        folder_path='./divided_log/B20170622102819_RT02486'
        x_data , y_data = get_trainingData(folder_path)





