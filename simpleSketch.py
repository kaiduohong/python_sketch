#-*- coding: UTF-8 -*-
import numpy as np
import sys
import logging as lg
def simpleSketch(A, sketchRowNum):
    if type(A) != np.matrixlib.defmatrix.matrix:
        raise TypeError('A is not a matrix, type = ' + str(type(A)))
    [n, m] = np.shape(A)
    B = np.matrix(np.zeros([sketchRowNum, m]))
    zeroRowNum = sketchRowNum
    iter = 0
    while iter < n:
        if iter % 100 == 0:
            print 'simpleSketch_sketchIter = '+ str(iter)

        [Null, S, V] = np.linalg.svd(B,full_matrices=False)
        #S = diag(S) numpy 返回的是array
        delta = np.power(S[int(np.floor((sketchRowNum + 1) / 2))],2)
        S = S**2 - delta
        S[S <=  sys.float_info.epsilon] = 0
        S = np.sqrt(S)
        B = np.matrix(np.diag(S )) * V
        zeroRowNum = sum(S <= sys.float_info.epsilon)

        index = min(iter + zeroRowNum, n)
        if iter + zeroRowNum <= n:
           B[sketchRowNum - zeroRowNum : sketchRowNum] = A[iter : index]
        else:
           B[sketchRowNum - zeroRowNum : sketchRowNum - (iter + zeroRowNum - n)] = A[iter : index]
        iter = iter + zeroRowNum + 1

    [Null, Null, B] = np.linalg.svd(B, full_matrices=False)
    return B.transpose()