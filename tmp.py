import numpy as np
import os
str='0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-9,382,391,96,979,969'
print 'ap',len(str.split(','))


str='0,0,0,0,0,0,0,0,0,0,2,0,0,0,1,0,0,0,1,0,1,0,1,1,0,4,383,386,87,980,969'
print 'ep',len(str.split(','))




def get_specifed_leaf(leaf_n=30 , dir_path='./divided_log/A20170614151153_RT02526 3'):
    f_ap=open(os.path.join(dir_path,'ap.txt'))
    f_ep=open(os.path.join(dir_path,'ep.txt'))

    ap_lines=f_ap.readlines()
    ep_lines=f_ep.readlines()

    assert len(ap_lines) == len(ep_lines)
    ap_ep_lines=zip(ap_lines , ep_lines)
    for i , ( ap_line , ep_line ) in enumerate( ap_ep_lines ):
        ap_line=np.asarray(ap_line.split(',')[:-1])
        ep_line=np.asarray(ep_line.split(',')[:-1])
        if i==0:
            ap_lines = ap_line
            ep_lines = ep_line
        else:
            ap_lines=np.vstack((ap_lines , ap_line))
            ep_lines=np.vstack((ep_lines, ep_line))
    #print ap_lines
    #print ep_lines
    return ap_lines[:, leaf_n] , ep_lines[:, leaf_n]


ap_ , ep_=get_specifed_leaf()
for i in range(len(ap_)):
    print float(ap_[i])-float(ep_[i])
print 'ap_',ap_[:30]
print 'ep_',ep_[:30]


