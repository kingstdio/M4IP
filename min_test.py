import minpy.numpy as np
import minpy.numpy.random as random
import numpy
import time
x= random.rand(1024, 1024)
y= random.rand(1024, 1024)
st = time.time()
for i in range(10):
    z= np.dot(x, y)
z.asnumpy()
print('time: {:.3f}.'.format(time.time()-st))


x= numpy.random.rand(1024, 1024)
y= numpy.random.rand(1024, 1024)
st = time.time()
for i in range(10):
    z= numpy.dot(x, y)
print('time: {:.3f}.'.format(time.time()-st))