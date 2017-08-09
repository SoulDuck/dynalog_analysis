import sys , os , glob

def show_processing(i,maxiter):
    msg='\r {} /{}'.format(i,maxiter)
    sys.stdout.write(msg)
    sys.stdout.flush()
