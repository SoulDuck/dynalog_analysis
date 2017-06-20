import numpy as np
import os , sys ,glob
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
    for path in log_paths:
        print path
        index=path.split('/')[-1].split('.')[0]
        print index
        f = open(path)
        lines=f.readlines()
        n_lines=len(lines)
        ele=lines[6].split(',')
        head_points=[];ep_=[];ap_=[];pfp_=[];nfp_=[]
        start_line=6
        end_line= n_lines
        for l in range(start_line , end_line):
            head_points.append(ele[0:14])
            for i in range(4):
                point_=ele[14 + 60 * i:14 + 60 * (i + 1)]
                if i % 4 == 0:
                    ep_.append(point_)
                elif i % 4 == 1:
                    ap_.append(point_)
                elif i % 4 == 2:
                    pfp_.append(point_)
                else:
                    nfp_.append(point_)
        head_points, ep_, ap_, pfp_, nfp_=map(np.asarray , [head_points , ep_ , ap_, pfp_  , nfp_])
        dic_log=list2dic(index,head_points, ep_, ap_, pfp_, nfp_)
        save_log(dic_log)
        #return index , head_points , ep_ , ap_, pfp_  , nfp_

if __name__ == '__main__':
    #list2dic()
    analysis_dinalog()
    #index, head_points, ep_, ap_, pfp_, nfp_=