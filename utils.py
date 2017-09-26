import sys , os , glob
import matplotlib
if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
def show_processing(i,maxiter):
    msg='\r {} /{}'.format(i,maxiter)
    sys.stdout.write(msg)
    sys.stdout.flush()

def plot_xy(test_predict , test_ys  , savename='./dynalog_result.png'):



    plt.plot(test_ys)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("leaf control point")
    plt.show()
    plt.savefig(savename)
    plt.close()

def make_dir(folder_name):
    if os.path.isdir(folder_name):
        return
    os.mkdir(folder_name)



