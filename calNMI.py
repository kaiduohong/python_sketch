import numpy as np
import KM
import nmi

def calNMI(olabels,plabels):
    if len(olabels) != len(plabels):
	raise Exception('len of labels not the same')
    if type(olabels) != np.ndarray:
	olabels = np.array(olabels)
    if type(plabels) != np.ndarray:
	plabels = np.array(plabels) 
    #calculate the weight
    weight = np.zeros(len(olabels)**2).reshape(len(olabels),len(olabels))
    for (i,L1) in enumerate(olabels):
	for (j,L2) in enumerate(plabels):
	   id1 = olabels == L1
	   id2 = plabels == L2 
	   id = id1 * id2
	   weight[i][j] = sum(id)
	    		
    km = KM.kuhn_Munkres(weight)
    mat = km.KM()
    olabels = mat[olabels]
    NMI = nmi.nmi(olabels,plabels)
    return NMI


if __name__ == '__main__':
    a=[2,2,2,2,2,2,3,3,3,3,3,3,1,1,1,1,1]
    b=[1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3]
    v = calNMI(a,b)
    print v
