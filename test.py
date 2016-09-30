import random as rd
from scipy.linalg.misc import norm
import numpy as np
# x = [rd.gauss(1000000,1000) for i in range(1000)]
# y = [rd.gauss(1000000,1000) for i in range(1000)]
# X = np.matrix([x,y]).transpose()
# [a,b,c] = np.linalg.svd(X,full_matrices=False)
# b = np.matrix(np.diag(b))
# print np.mean(X,0)
# X = X - np.mean(X,0)
# plot(X)
# [a,b,d] = np.linalg.svd(X,full_matrices=False)
# c = np.matrix(c)
# d = np.matrix(d)
#c = c[0]
#d = d[0]

#print c.transpose() * c
#print norm(c.transpose()*c-  d.transpose() * d,'fro') / norm(c,'fro')
b = np.array([[[1, 2, 3, 4],
        [1, 2, 3, 4]],

       [[1, 2, 3, 4],
        [1, 2, 3, 4]]])
c = {}
c['nother'] = []
c['nother'].append(b)
c['nother'].append(b)
c['nother'].append(b)
d = b
d[0] = [[12,6,7,3],[9,5,3,5]]
print np.shape(d)
c['nother'][0] = d

