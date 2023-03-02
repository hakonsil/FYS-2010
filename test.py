import numpy as np
import scipy
"""TASK 3"""
w = np.zeros((3, 3))
w[0, 2] = 1
w[1, 1] = 1
w[2, 0] = 1
w[1,2] = 2
w[2,1] = 2
w[2,2] = 3

f = np.zeros((3, 3))
f[0,1] = 1
f[1,1] = 1
f[2,0] = 1

#convolve f and w

g = scipy.signal.convolve2d(w, f, mode='same', boundary='fill', fillvalue=0) # zero padding
print(g)
q = scipy.signal.convolve2d(f, w, mode='same', boundary='wrap') # cyclic convolution
print(q)

