# -*- coding:utf8 -*-
import numpy as np
import sys
import math


def nmi(A, B): #两个列
    if len(A) != len(B):
        raise Exception("lenA not eq lenB") 

    total = float(len(A))
    A_ids = np.unique(A)
    B_ids = np.unique(B)

    #Mutual information
    MI = 0
    for idA in A_ids:
	for idB in B_ids:
	    idAOccur = A == idA
	    idBOccur = B == idB
	    idABOccur = idAOccur * idBOccur
            
            px = sum(idAOccur) / total
	    py = sum(idBOccur) / total
	    pxy = sum(idABOccur) / total 

	    MI = MI + pxy * math.log(pxy / (px * py) + sys.float_info.epsilon,2)

    Hx = 0
    for idA in A_ids:
	idAOccurCount = sum( A == idA) 
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount /total + sys.float_info.epsilon,2)

    Hy = 0
    for idB in B_ids:
        idBOccurCount = sum(B == idB)
	Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + sys.float_info.epsilon,2)

    if Hx == 0 and Hy == 0:
	return 1
    return 2 * MI / (Hx + Hy)
        
if __name__=='__main__':
    a=[1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3]
    b=[1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3]
    v = nmi(a,b)
    print v
