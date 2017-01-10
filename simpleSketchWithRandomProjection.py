#-*-coding:utf-8-*-

def simpleSketch_RandomProjection(samples,sketchRowNum,mu,sigma):
    if type(A) != np.matrixlib.defmatrix.matrix:
        raise TypeError('A is not a matrix, type = ' + str(type(A)))
    [m,n] = np.shape(A)
    gaussMatrix = np.random.normal(0,1,size=(n,sketchRowNum))
   
    iter = 0
    patch = 1000
    while iter < m:
        if iter % 5000 == 0:
            print 'simpleSketch_RandomProjection'+str(iter)
         
	  

