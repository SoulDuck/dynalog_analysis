import numpy as np
import os , sys ,glob
import pickle
import scipy.io as sio
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
        for l in range(start_line , end_line):
            leafs = []
            elements=lines[l].split(',')
            head_info.append(elements[0:14])
            try:
                if len(elements) != 254:
                    raise ValueError
            except ValueError:
                print 'we skip this line',l, ' to save because',len(elements)
                continue
            for i in range(n_leafs):
                points_=elements[14 +( i * 4 ) :14+ ( i + 1 )*4]
                leafs.append(points_)
            leafs_matrix.append(leafs)
        f=open('./divided_log/'+index+'/leaf_txt','wb')
        pickle.dump(list(leafs_matrix) ,f )
        f.close()
        f = open('./divided_log/' + index + '/head_txt','wb')
        pickle.dump(list(head_info),f)
        f.close()

        np.save('./divided_log/'+index+'/leaf',leafs_matrix)
        np.save('./divided_log/' + index + '/head', head_info)
        sio.savemat('./divided_log/'+index+'/leaf', {'leaf': leafs_matrix})
        sio.savemat('./divided_log/' + index + '/head', {'head': head_info})

        print np.shape(leafs_matrix)




        #head_points, ep_, ap_, pfp_, nfp_=map(np.asarray , [head_points , ep_ , ap_, pfp_  , nfp_])
        #dic_log=list2dic(index,head_points, ep_, ap_, pfp_, nfp_)
        #save_log(dic_log)
        #return index , head_points , ep_ , ap_, pfp_  , nfp_

if __name__ == '__main__':
    #list2dic()
    analysis_dinalog()
    #ap_=np.load('./divided_log/A20170614151153_RT02526/leaf.npy')
    f=open('./divided_log/A20170614151153_RT02526/head_txt')
    a=pickle.load(f)
    print np.shape(a)