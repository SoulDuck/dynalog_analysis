import numpy as np
a=np.zeros([1,1])
b=np.zeros([1,1])
c=np.zeros([1,1])
#b=2
#c=3
print type(a).__module__
print np.__name__
def abc(*args):
    print len(args)
    for arg in args:
        print arg

    return args
print abc(a,b,c)

a=[1,1,1,1,1,1,1,1,1,5]
print a[:None]