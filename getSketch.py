#-*- coding: UTF-8 -*-
import numpy as np
from belongWhichPieces import *
def getSketch(A,Pieces,numberOfPieces,sketchNum):
#按行存储
    if type(A) != np.matrixlib.defmatrix.matrix:
        raise TypeError('A is not a matrix, type = ' + str(type(A)))

    [sampleNum , Null] = np.shape(A);
    sketchMatrix = np.matrix(np.zeros([sampleNum,sketchNum]))
    rightSubSpace = Pieces['rightSubspace']

    for i in range(sampleNum):
        index = belongWhichPieces(A[i], Pieces, numberOfPieces)
        U = rightSubSpace[index]
        sketchMatrix[i] = A[i] * U.transpose()
    return sketchMatrix