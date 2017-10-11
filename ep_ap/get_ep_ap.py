import numpy as np
import matplotlib.pyplot as plt
folders=['A20170614161427_RT02486','A20170615085606_RT02473','A20170615083340_RT02494']
def plot_xy(test_predict , test_ys  , savename='./dynalog_result.png'):



    plt.plot(test_ys)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("leaf control point")
    plt.show()
    plt.savefig(savename)
    plt.close()

def get_ep_specify_leaf(num_leaf , folders):
    count=0
    f_out = open('./' + str(num_leaf) + '_ep.txt', 'w')
    for f in folders:
        f = open(f + '/ep.txt', 'r')
        lines=f.readlines()
        for line in lines:
            #print line.split(',')[30]
            f_out.write(line.split(',')[30]+',')
            count+=1
    print count
get_ep_specify_leaf(30 , folders)
def get_ap_specify_leaf(num_leaf,folders):
    count=0
    f_out = open('./' + str(num_leaf) + '_ap.txt', 'w')
    for f in folders:
        f = open(f + '/ap.txt', 'r')
        lines = f.readlines()
        for line in lines:
            #print line.split(',')[30]
            f_out.write(line.split(',')[30] + ',')
            count+=1
    print count

get_ap_specify_leaf(30, folders)
def get_ap_hat_specify_leaf(num_leaf,predict_np):
    np.load(predict_np)
    f_out = open('./' + str(num_leaf) + '_ap_hat.txt', 'w')
    for e in np.load(predict_np):
        #print int(e)
        f_out.write(str(int(e))+',')

get_ap_hat_specify_leaf(30,'true_original.npy')
#f_true=open('30_ap.txt' ,'r')
f_true=open('30_ap_hat.txt' ,'r')
f_pred=open('30_ap_hat_.txt' ,'r')
ele_true=f_true.readline().split(',')[:-1]
ele_pred=f_pred.readline().split(',')[:-1]
assert len(ele_true) == len(ele_pred)
for i,ele in enumerate(ele_true):
    count=0
    if ele==ele_pred[i]:
        count+=1
print count/float(i)
plot_xy(ele_true , ele_pred)