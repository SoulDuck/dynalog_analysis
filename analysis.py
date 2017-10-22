#-*- coding -*-
import numpy as np
import os , sys ,glob
import pickle
import scipy.io as sio
import matplotlib
if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import data

debug_lv0=True
debug_lv1=False

def get_rtfile(folder='./log/'):
    #dynalog_path = 'xxxx_rt0000'
    paths=glob.glob(folder+'/*.dlg')
    def extract_rtname(path):
        rtname=path.split('_')[-1].split('.dlg')[0]
        return rtname
    names=map(extract_rtname,paths)
    print len(names)
    names=set(names)
    print len(names)
    f=open('./rt_names.txt','w')
    for name in names:
        f.write(name)
        f.write('\n')
    return names
def extract_points(folder_path):
    log=np.load(folder_path+'leaf.npy')
    log_head = np.load(folder_path + 'head.npy')
    f_head=open(folder_path+'head.txt','w')
    fs=[]
    fs.append(open(folder_path + '/ep.txt' , 'w'))
    fs.append(open(folder_path + '/ap.txt' , 'w'))
    fs.append(open(folder_path + '/pfp.txt','w'))
    fs.append(open(folder_path + '/nfp.txt','w'))
    n_lines,n_leaf,n_points=np.shape(log)

    for p_ind,f in enumerate(fs):
        #print 'a' ,np.shape(log[:, :, p_ind])
        lines=log[:, :, p_ind] #lines shape = (1490 , 60)
        for line in lines: #line shape (60)
            for ele in line:
                f.write(ele+',')
            f.write('\n')
        f.close()
    for head in log_head:
        for ele in head:
            f_head.write(ele+'\t')


        f_head.write('\n')

def save_file(filepath ,leafs_matrix , head_info ):
    ##pickle
    f = open(os.path.join(filepath,'leaf.pkl'), 'wb')
    pickle.dump(list(leafs_matrix), f)
    f.close()
    f = open(os.path.join(filepath, 'head.pkl'), 'wb')
    pickle.dump(list(head_info), f)
    f.close()
    ##numpy
    np.save(os.path.join(filepath,'leaf'), leafs_matrix)
    np.save(os.path.join(filepath,'head'), head_info)

    ##matlab
    sio.savemat(os.path.join(filepath,'leaf'), {'leaf': leafs_matrix})
    sio.savemat(os.path.join(filepath,'head'), {'head': head_info})


def list2dic(index, header, ep_ , ap_ , pfp_ , npf_):
    return_dic={}
    return_dic['index']=index
    return_dic['header']=header
    return_dic['ep'] = ep_
    return_dic['ap']=ap_
    return_dic['pfp']=pfp_
    return_dic['nfp']=npf_
    return return_dic
def save_log(logs):
    folder_path='./divided_log/'+logs['index']+'/'
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    for key in logs.keys():
        np.save(folder_path+key,logs[key])
def analysis_dinalog():
    log_paths=glob.glob('./log/*.dlg')
    print log_paths[0]
    for path in log_paths:
        print path
        index=path.split('/')[-1].split('.')[0] #
        if not os.path.isdir('./divided_log/'+index):
            os.mkdir('./divided_log/'+index)
        f = open(path)
        lines=f.readlines()
        n_lines=len(lines)
        start_line=6 # why lines counter start at 6?
        end_line= n_lines
        head_info=[];leafs_matrix=[]
        n_leafs=60
        ap_=[];ep_=[];pfp_=[];nfp_=[];
        for l in range(start_line , end_line):
            leafs = []
            elements=lines[l].split(',')
            head_info.append(elements[0:14]) # all lines heads was included leafs_matrix
            try:
                if len(elements) != 254:
                    raise ValueError
            except ValueError:
                print 'we skip this line',l+1, 'because the # line elements ',len(elements) ,'not 254', 'file_name',index
                continue
            for i in range(n_leafs):
                points_=elements[14 +( i * 4 ) :14+ ( i + 1 )*4]
                leafs.append(points_)
            leafs_matrix.append(leafs) # all lines leafs was included leafs_matrix
        save_file(os.path.join('./divided_log/'+index ),leafs_matrix , head_info) #
        extract_points('./divided_log/'+index+'/') #

        #head_points, ep_, ap_, pfp_, nfp_=map(np.asarray , [head_points , ep_ , ap_, pfp_  , nfp_])
        #dic_log=list2dic(index,head_points, ep_, ap_, pfp_, nfp_)
        #save_log(dic_log)
        #return index , head_points , ep_ , ap_, pfp_  , nfp_

    """
    :param ep: txt file  
    :param ap: txt file 
    :param ep_larger: numpy  
    :param ep_same: numpy 
    :param ep_less: numpt 
    :return: 
    """






def get_acc(true , pred  , error_range_percent):
    assert len(true) == len(pred)
    true_count = 0;
    for i, v in enumerate(true):
        if true[i] - true[i] * (error_range_percent / 100.) <= pred[i] and pred[i] <= true[i] + true[i] * (
            error_range_percent / 100.):
            true_count += 1
    acc=true_count/float(len(pred))
    print 'accuracy :' ,acc, 'error_range : ' , error_range_percent
    return acc


def get_acc_with_ep(ep ,true , pred  , error_range_percent):
    assert np.max(pred) >=1.

    if debug_lv0:
        print 'analysis.py | get_acc_with_ep '
    assert len(true) == len(pred)
    true_count = 0;
    """type 1 of gettting accuracy """
    """
    for i, v in enumerate(true):
        
        diff=ep[i]-true[i]
        up_range=true[i]+diff*(error_range_percent / 100.)
        buttom_range=true[i]-diff*(error_range_percent / 100.)

    if debug_lv1:
        if diff is not 0:
            print 'diff ', diff
    """
    """type 2 of gettting accuracy """
    for i, v in enumerate(true):
        up_range = true[i] + error_range_percent
        buttom_range = true[i] - error_range_percent
        #print up_range
        #print buttom_range
        if buttom_range<= pred[i] and pred[i] <= up_range:
            true_count += 1
    acc=true_count/float(len(pred))
    print 'accuracy :' ,acc, 'error_range : ' , error_range_percent
    return acc




def analysis_result(ep , true , pred  , error_range_percent):
    """
    :param true: type must be numpy
    :param pred: type must be numpy
    :param error_range_percent: if error_range_percent = 5 --> 5%
    :return:
    """
    true=np.squeeze(true)
    pred = np.squeeze(pred)
    print np.shape(true)
    print np.shape(pred)
    plt.figure(figsize=(30, 30))
    red_patch = mpatches.Patch(color='red', label='True')
    blue_patch = mpatches.Patch(color='blue', label='False')
    plt.legend(handles=[red_patch , blue_patch])
    true_count=0;

    for i, v in enumerate(true):
        diff = abs(ep[i]-v)
        up_range = true[i] + diff* (error_range_percent / 100.)
        down_range = true[i] - diff* (error_range_percent / 100.)


        if up_range >= pred[i] and pred[i] >= down_range:
            plt.scatter(i , true[i] , c ='r' , label = 'True ap')
            true_count +=1
        else :
            plt.scatter(i , true[i]  , c='b' , label = 'False ap')
    acc = true_count / float(len(pred))
    print 'accuracy :', acc, 'error_range : ', error_range_percent
    plt.savefig('./graph/result_analysis.png')
    plt.show()


if __name__ == '__main__':
    #list2dic()
    #print get_rtfile()
    analysis_dinalog()
    #ap_=np.load('./divided_log/A20170614151153_RT02526/leaf.npy')
    #leaf=np.load('./divided_log/A20170614151153_RT02526/leaf.npy')
    #a=leaf[:,:,0]
    #extract_points('./divided_log/A20170614151153_RT02526/' )
