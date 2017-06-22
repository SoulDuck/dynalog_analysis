import numpy as np
import os , sys ,glob
import pickle
import scipy.io as sio

def extract_points(folder_path):
    log=np.load(folder_path+'leaf.npy')
    log_head = np.load(folder_path + 'leaf.npy')

    f_head=open(folder_path+'head.txt','w')
    fs=[]
    fs.append(open(folder_path+'/ap.txt' , 'w'))
    fs.append(open(folder_path + '/ep.txt' , 'w'))
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
            f_head.write(ele)





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
        index=path.split('/')[-1].split('.')[0]
        if not os.path.isdir('./divided_log/'+index):
            os.mkdir('./divided_log/'+index)
        f = open(path)
        lines=f.readlines()
        n_lines=len(lines)
        start_line=6
        end_line= n_lines
        head_info=[];leafs_matrix=[]
        n_leafs=60
        ap_ =[];ep_ =[];pfp_ =[];nfp_ =[];
        for l in range(start_line , end_line):
            leafs = []
            elements=lines[l].split(',')
            head_info.append(elements[0:14])
            try:
                if len(elements) != 254:
                    raise ValueError
            except ValueError:

                print 'we skip this line',l, ' to save because',len(elements) , 'file_name',index
                continue
            for i in range(n_leafs):
                points_=elements[14 +( i * 4 ) :14+ ( i + 1 )*4]
                leafs.append(points_)
            leafs_matrix.append(leafs)
        save_file(os.path.join('./divided_log/'+index ),leafs_matrix , head_info)
        extract_points('./divided_log/'+index+'/')




        #head_points, ep_, ap_, pfp_, nfp_=map(np.asarray , [head_points , ep_ , ap_, pfp_  , nfp_])
        #dic_log=list2dic(index,head_points, ep_, ap_, pfp_, nfp_)
        #save_log(dic_log)
        #return index , head_points , ep_ , ap_, pfp_  , nfp_

if __name__ == '__main__':
    #list2dic()
    analysis_dinalog()
    #ap_=np.load('./divided_log/A20170614151153_RT02526/leaf.npy')
    #leaf=np.load('./divided_log/A20170614151153_RT02526/leaf.npy')
    #a=leaf[:,:,0]
    #extract_points('./divided_log/A20170614151153_RT02526/' )
