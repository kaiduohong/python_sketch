#-*- coding: UTF-8 -*-
import numpy as np
import sys
from belongWhichPieces import *
from scipy.linalg.misc import norm
def sketchTrial(data, sketchNum, threadhold):
    if type(data) != np.matrixlib.defmatrix.matrix:
        raise TypeError('data is not a matrix, type = ' + str(type(data)))

    sketchNum = sketchNum * 2
    [dataNum,dataDim] = np.shape(data)
    index = np.zeros(dataNum)

    #alpha = 2
    #beta = 0.01

    numberOfPieces = 0


    Pieces = {}
    Pieces['rightSubspace'] = []
    dataTemp = []

    numberOfZeroRows = []
    
    for i in xrange(dataNum):
        if i % 1000 == 0:
            print 'sketchTrial_sketchIter = '+ str(i + 1)

        belongsId = belongWhichPieces(data[i], Pieces, numberOfPieces, threadhold)

        if belongsId == -1:
            dataTemp.append(np.zeros([sketchNum,dataDim]))
	    np.matrix(np.zeros([sketchNum,dataDim])) 
 	    u = np.matrix(np.zeros([sketchNum,dataDim]))
            u[0] = data[i]
            numberOfZeroRows.append(sketchNum - 1)
            Pieces['rightSubspace'].append(u / norm(u,'fro'))
            dataTemp.append(u)
            numberOfPieces += 1
        else:
            zerosRowNum = numberOfZeroRows[belongsId]
            dataTemp[belongsId][sketchNum - zerosRowNum] = data[i]
            #Pieces['rightSubspace'][belongsId][sketchNum - zerosRowNum] = data[i]
            zerosRowNum = numberOfZeroRows[belongsId]
            zerosRowNum = zerosRowNum - 1
            if zerosRowNum == 0:
                B = dataTemp[belongsId]
                [Null,s,u] = np.linalg.svd(B,full_matrices=False)
                delta = np.power(s[int(np.floor((sketchNum + 1) / 2))], 2)
                s = s**2 - delta
                s[s <= sys.float_info.epsilon] = 0
                s = np.sqrt(s)
                zerosRowNum = sum(s <= sys.float_info.epsilon)
                B = np.matrix(np.diag(s)) * u
                Pieces['rightSubspace'][belongsId] = u[:sketchNum]
                dataTemp[belongsId] = B[:sketchNum]

	    numberOfZeroRows[belongsId] = zerosRowNum


    for i in range(numberOfPieces):
        B = dataTemp[i]#Pieces['rightSubspace'][i]
	[Null,s,B]  = np.linalg.svd(B,full_matrices=False)
        zerosRowNum = sum(s <= sys.float_info.epsilon)
        Pieces['rightSubspace'][i] = B[0:sketchNum/2]

    return Pieces,numberOfPieces,index
