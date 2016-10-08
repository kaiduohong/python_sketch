#-*- coding: UTF-8 -*-
import numpy as np
import sys
from belongWhichPieces import *
from scipy.linalg.misc import norm
def sketchTrial(data, sketchNum, threadhold):
    if type(data) != np.matrixlib.defmatrix.matrix:
        raise TypeError('data is not a matrix, type = ' + str(type(data)))

    [dataNum,dataDim] = np.shape(data)
    index = np.zeros(dataNum)

    #alpha = 2
    #beta = 0.01

    numberOfPieces = 0


    Pieces = {}
    Pieces['rightSubspace'] = []
    Pieces['sigularValue'] = []

    for i in range(dataNum):
        if i % 1000 == 0:
            print 'sketchTrial_sketchIter = '+ str(i + 1)

        belongsId = belongWhichPieces(data[i], Pieces, numberOfPieces, threadhold)

        if belongsId == -1:
            numberOfPieces = numberOfPieces + 1

            u = np.matrix(np.zeros([sketchNum,dataDim]))
            u[0] = data[i] / norm(data[i] , 2)

            s = np.matrix(np.zeros([sketchNum,sketchNum]))
            s[0,0] = norm(data[i] , 2)

            Pieces['rightSubspace'].append(u)
            Pieces['sigularValue'].append(s)
        else:
            u = Pieces['rightSubspace'][belongsId]
            s = Pieces['sigularValue'][belongsId]
            s = np.diag(s)
            delta = np.power(s[int(np.floor((sketchNum + 1) / 2))], 2)
            s = s**2 - delta
            s[s <= sys.float_info.epsilon] = 0
            s = np.sqrt(s)
            #print 'sss = ',np.shape(s),np.shape(np.diag(s))
            # print 'ddd = ',np.diag(s)
            zeroRowNum = sum(s <= sys.float_info.epsilon)

            v = np.matrix(np.diag(s)) * u
            v[sketchNum - zeroRowNum] = data[i]
            [Null,s,u] = np.linalg.svd(v,full_matrices=False)

            Pieces['rightSubspace'][belongsId]=u[0:sketchNum]
            #print 'shape ',np.shape(Pieces['sigularValue'][belongsId]),' ss ',np.shape(s[0:sketchNum][0:sketchNum])
            Pieces['sigularValue'][belongsId] = np.matrix(np.diag(s[0:sketchNum]))

    return Pieces,numberOfPieces,index
