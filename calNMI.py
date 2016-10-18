import numpy as np
import KM
import nmi

def calNMI(plabels,olabels):
    if len(olabels) != len(plabels):
	raise Exception('len of labels not the same')
    if type(olabels) != np.ndarray:
	olabels = np.array(olabels)
    if type(plabels) != np.ndarray:
	plabels = np.array(plabels) 

    #calculate the weight
    uniqueOlabels = np.unique(olabels)
    uniquePlabels = np.unique(plabels)
    
    uOLen = len(uniqueOlabels)
    uPLen = len(uniquePlabels)
    for (i,uo) in enumerate(uniqueOlabels):
	olabels[olabels == uo] = i
    for (i,up) in enumerate(uniquePlabels):
	plabels[plabels == up] = i

	
    
    uniqueOlabels = np.unique(olabels)
    uniquePlabels = np.unique(plabels)

    weight = np.zeros(uOLen * uPLen).reshape(uPLen,uOLen)
    for (i,L1) in enumerate(uniquePlabels):
	for (j,L2) in enumerate(uniqueOlabels):
	   id1 = olabels == L1
	   id2 = plabels == L2 
	   id = id1 * id2
	   weight[i][j] = sum(id)
	    		
    km = KM.kuhn_Munkres(weight)
    mat = km.KM()
    plabels = mat[plabels]
    NMI = nmi.nmi(plabels,olabels)
    return NMI


if __name__ == '__main__':
    a=[1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3]
    b=[1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3]
    v = calNMI(a,b)
    print v
