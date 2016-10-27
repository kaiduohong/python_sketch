#-*- coding: UTF-8 -*-
have not complete
import numpy as np
import sys
from belongWhichAff import *
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
    Pieces['bias'] = []
    dataTemp = []

    numberOfZeroRows = []

    for i in range(dataNum):
        if i % 1000 == 0:
            print 'sketchTrial_sketchIter = '+ str(i + 1)

        belongsId = belongWhichAff(data[i], Pieces, numberOfPieces, threadhold)

        if belongsId == -1:
            u = np.matrix(np.zeros([sketchNum,dataDim]))
            u[0] = data[i]
	    Pieces['bias']['num'].append(1)
	    Pieces['bias']['aff'].append(u)
            numberOfZeroRows.append(sketchNum - 1)
            dataTemp.append(u)
            Pieces['rightSubspace'].append(u / norm(u,'fro'))
            numberOfPieces = numberOfPieces + 1
        else:
	    dataTemp[sketchNum - zerosRowNum] = data[i]
 	    #Pieces['rightSubspace'][belongsId][sketchNum - zerosRowNum] = data[i]
	    n = Pieces['bias']['num']
	    Pieces['bias']['aff'] = Pieces['bias']['aff'] * (n / float(n + 1)) + data[i] / float(n + 1)
	
            zerosRowNum = numberOfZeroRows[belongsId]
            zerosRowNum = zerosRowNum - 1
            if zerosRowNum == 0:
                B = dataTemp[belongsId]#Pieces['rightSubspace'][belongsId]
                [Null,s,u] = np.linalg.svd(B,full_matrices=False)
                delta = np.power(s[int(np.floor((sketchNum + 1) / 2))], 2)
                s = s**2 - delta
                s[s <= sys.float_info.epsilon] = 0
                s = np.sqrt(s)
                zerosRowNum = sum(s <= sys.float_info.epsilon)
                B = np.matrix(np.diag(s)) * u
                Pieces['rightSubspace'][belongsId] = u[:sketchNum]
                dataTemp = B[:sketchNum]

	    numberOfZeroRows[belongsId] = zerosRowNum

    for i in range(numberOfPieces)
        B = dataTemp[i]
	[Null,s,B]  = np.linalg.svd(B,full_matrices=False)
        zerosRowNum = sum(s <= sys.float_info.epsilon)
        Pices['rightSubspace'][i] = B[0:sketchNum - zerosRowNum]

    return Pieces,numberOfPieces,index
