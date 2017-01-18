#-*- coding: UTF-8 -*-
#update date:20170107
import numpy as np
import sys
import logging as lg

def getRandomWeightSimpleSketchData(samples,testSamples,sketchM):
    sketchTrainData = samples * sketchM
    sketchTestData = testSamples * sketchM
    return sketchTrainData,sketchTestData
    
def randomWeightSimpleSketch(A, sketchRowNum,proportion, weight = 1):
    if type(A) != np.matrixlib.defmatrix.matrix:
        raise TypeError('A is not a matrix, type = ' + str(type(A)))
    index = random.sample(range(n),int(np.floor(n * proportion)))
    A = A[index]
    sketchRowNum = sketchRowNum * 2
    [n, m] = np.shape(A)
    subspace = np.matrix(np.random.normal([sketchRowNum / 2,m]))
    B = np.matrix(np.zeros([sketchRowNum, m]))
    zeroRowNum = sketchRowNum
    iter = 0
    while iter < n:
        if iter % 1000 == 0:
            print 'simpleSketch_sketchIter = '+ str(iter)
	
        [Null, S, V] = np.linalg.svd(B,full_matrices=False)
        delta = S[int(np.floor((sketchRowNum + 1) / 2))]**2
        S = S**2 - delta
        S[S <=  sys.float_info.epsilon] = 0
        S = np.sqrt(S)
	B = np.matrix(np.multiply(S,V.transpose()).transpose())
        zeroRowNum = sum(S <= sys.float_info.epsilon)

        index = min(iter + zeroRowNum, n)
        if iter + zeroRowNum <= n:
           B[sketchRowNum - zeroRowNum : sketchRowNum] = weight * A[iter : index]
        else:
           B[sketchRowNum - zeroRowNum : sketchRowNum - (iter + zeroRowNum - n)] = weight * A[iter : index]
 	subspace += np.random.normal([sketchNum / 2,index - iter]) * A[iter : index]	
        iter = iter + zeroRowNum + 1
 
    [Null, Null, B] = np.linalg.svd(B, full_matrices=False)
    subspace = subspace * B.transpose()
    B = B[:sketchRowNum / 2]
    [Q,Null] = np.liang.qr(subspace.transpose(),mode='economic')
    B = np.concatenate(B.transpose(),Q),1)
    return B
