#-*- coding: UTF-8 -*-
#update date:20170107
import numpy as np
import sys
import logging as lg

#产生降维数据
def getSimpleSketchData(samples,testSamplesmsketchM):
    sketchTrainData = samples * sketchM
    sketchTestData = testSamples * sketchM
    return sketchTrainData,sketchTestData
    
    
#这是原始的simpleSketch 算法
def simpleSketch(A, sketchRowNum):
    #输入为矩阵，以行排列
    if type(A) != np.matrixlib.defmatrix.matrix:
        raise TypeError('A is not a matrix, type = ' + str(type(A)))
    #这是最后要降成一半，为了使得效果不那么差，先乘以2
    sketchRowNum = sketchRowNum * 2
    [n, m] = np.shape(A)
    B = np.matrix(np.zeros([sketchRowNum, m]))
    zeroRowNum = sketchRowNum
    iter = 0
    while iter < n:
        if iter % 1000 == 0:
            print 'simpleSketch_sketchIter = '+ str(iter)
	
        [Null, S, V] = np.linalg.svd(B,full_matrices=False)
        #S = diag(S) numpy 返回的是array
	#取中间的值
        delta = S[int(np.floor((sketchRowNum + 1) / 2))]**2
        S = S**2 - delta
        S[S <=  sys.float_info.epsilon] = 0
        S = np.sqrt(S)
	#
        #B = np.matrix(np.diag(S)) * V
	#改成广播的形式,以列的方式就没那么麻烦
	B = np.matrix(np.multiply(S,V.transpose()).transpose())
        zeroRowNum = sum(S <= sys.float_info.epsilon)

        index = min(iter + zeroRowNum, n)
        if iter + zeroRowNum <= n:
           B[sketchRowNum - zeroRowNum : sketchRowNum] = A[iter : index]
        else:
           B[sketchRowNum - zeroRowNum : sketchRowNum - (iter + zeroRowNum - n)] = A[iter : index]
        iter = iter + zeroRowNum + 1
 

    [Null, Null, B] = np.linalg.svd(B, full_matrices=False)
    #zeroRowNum = sum(S <= sys.float_info.epsilon)
    B = B[:sketchRowNum/2]
    
    return B.transpose()
