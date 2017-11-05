import numpy as np
import os , sys , glob
#ep-ap error



def get_trainingData(folder_path , seq_length=7 , seq_width=3 ,save_is=True):
    debug_flag = True
    debug_flag_lv1 = True
    debug_flag_lv2 = False
    if __debug__ == debug_flag:
        print '### debug | preprocessing.py | get_trainingData ###'
        print 'hidden param state: seq_length=',seq_length,'save_is=',save_is
        print 'folder path=',folder_path

    #type(ep).__moduel__ == __name__.np

    leafs = np.load(os.path.join(folder_path,'leaf.npy'))
    heads = np.load(os.path.join(folder_path, 'head.npy'))


    n_length , n_leaf , n_param = np.shape(leafs)
    head_height, head_width= np.shape(heads)
    print 'leafs shape : n_length: {} n_leaf: {} n_param :{} '.format(n_length , n_leaf , n_param) # ( 1489 60 4 )
    print 'header shape : head height : {} head width : {}'.format(head_height, head_width)  # ( 1489 14  )
    ep_ = leafs[:,:,0]#1489 60
    ap_ = leafs[:,:,1]#1489 60

    x_data=np.zeros([n_length-seq_length-1 , n_leaf-seq_width+1 , seq_length , seq_width])
    x_BCG_data=np.zeros([n_length-seq_length-1 , n_leaf-seq_width+1 , seq_length , seq_width+3]) # +3 --> BeamOfOff , Colimator , Gantry
    y_data=np.zeros([n_length-seq_length-1 , n_leaf-seq_width+1])
    y1_data = np.zeros([n_length - seq_length - 1, n_leaf - seq_width + 1])

    for r in range(n_length-seq_length-1):
        x=ep_[r:r+seq_length] #(7,60)
        y=ap_[r+seq_length] #(7,60)
        y1= ap_[r + seq_length+1] #(7,60)
        head_x=heads[r:r+seq_length]

        for c in range(n_leaf-seq_width+1):
            x_data[r, c,:,:]=x[:,c:c+seq_width]
            x_BCG_data[r,c,:,:seq_width]=x[:,c:c+seq_width]
            x_BCG_data[r, c, :, seq_width :]
            x_BCG_data[r, c, :, seq_width:seq_width + 1] = head_x[:, 3:4]
            x_BCG_data[r, c, :, seq_width+1:seq_width + 2] = head_x[:, 6:7]
            x_BCG_data[r, c, :, seq_width + 2:seq_width + 3] = head_x[:, 7:8]
            y_data[r,c]=y[c+1]
            y1_data[r, c] = y1[c + 1]

    print 'x with B C G data shape : {} '.format(np.shape(x_BCG_data))
    #print x_BCG_data[0, 0, :, 3]
    #print x_BCG_data[1, 0, :, 3]
    #print x_BCG_data[0, 1, :, 3]
    #print heads[:14, 3:4]



    np.save(os.path.join(folder_path, 'x_data_BCG.npy'), x_BCG_data)
    np.save(os.path.join(folder_path ,'x_data.npy'), x_data)
    np.save(os.path.join(folder_path, 'y_data.npy'), y_data)
    np.save(os.path.join(folder_path, 'y1_data.npy'), y1_data)
    assert len(x_data) == len(y_data) == len(y1_data)
    if __debug__ == debug_flag_lv1:
        print 'x_data , and y_data was saved'
    if __debug__ == debug_flag_lv2:
        print 'x  data length :,',np.shape(x_data)
        print 'y  data length :,',np.shape(y_data)
        print 'y1  data length :,', np.shape(y1_data)
        print 'leaf shape :',np.shape(leafs)
        print 'ap shape :', np.shape(ep_)
        print 'ep shape :', np.shape(ap_)
        print '###check###'
        print y1_data[0]
        print y_data[0]
        print x_data[0]
    return x_data , y_data , y1_data


if __name__ =='__main__':
    root_dir='./divided_log'
    path, subfolders , files=os.walk(root_dir).next()
    print '# subfolders ' , len(subfolders)
    for subfolder in subfolders:
        target_folder_path=os.path.join('./divided_log' , subfolder)
        x_data , y_data, y1_data = get_trainingData(target_folder_path)

