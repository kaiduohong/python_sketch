#-*- coding:utf8 -*-
import sys
import numpy as np
class kuhn_Munkres(object):
#weight 是二分图的边的权重，是一个矩阵（二维数组）
#nodeNum 节点数(顶点数目vertexNum）
#Lx Ly 
#left[i] 是右边第i个点匹配的点的编号 未匹配的标-1
#s[] t[] 做右边匈牙利树节点标记
#slack[i] 延迟标记


    def __init__(self, weight):
	[n,m] = np.shape(weight)
	if n > m:
	    raise Exception('n > m')
	self.leftNodeNum = n
	self.rightNodeNum = m
	self.weight = np.array(weight,dtype = int)
	self.Lx = np.array(np.zeros(n),dtype = int)
	self.Ly = np.array(np.zeros(m),dtype = int)
	self.left = np.array(np.zeros(m),dtype = int)
 	self.slack = np.array(np.zeros(m),dtype=int)
	self.s = np.array(np.zeros(n),dtype = bool)
	self.t = np.array(np.zeros(m),dtype = bool)
	 
    def init(self):
	for y in xrange(self.rightNodeNum):
	    self.left[y] = -1
	    self.Ly[y] = 0
	#self.Ly[:] = 0
	#self.left[:] = -1
	#self.Lx[:] = 0
	for i in xrange(self.leftNodeNum):
	    self.Lx[i] = 0
	    for j in xrange(self.rightNodeNum):
		self.Lx[i] = np.max([self.Lx[i],self.weight[i][j]])
	for y in xrange(self.rightNodeNum):
	    for x in xrange(self.leftNodeNum):
		self.slack[y] = np.min([self.Lx[x] - self.weight[x][y],self.slack[y]])
	    
    def match(self,x):
	self.s[x] = True
	for y in xrange(self.rightNodeNum):
	    if not self.t[y]:
	        if self.Lx[x] + self.Ly[y] == self.weight[x][y]:
		    self.t[y] = True
		    if self.left[y] == -1:
			self.left[y] = x
			return True
		    elif self.match(self.left[y]):
			self.left[y] = x
			return True
		else:
		    self.slack[y] = min([self.slack[y], \
			self.Lx[x] + self.Ly[y] - \
			self.weight[x][y]])

    def update(self):
	a = sys.maxint
	for y in xrange(self.rightNodeNum):
	    if not self.t[y]:
		a = min([a,self.slack[y]])
	a = int(np.round(a))
	for i in xrange(self.leftNodeNum):
	    if self.s[i]:
		self.Lx[i] -= a
	    if self.t[i]:
		self.Ly[i] += a

    def KM(self):
 	self.init()	
	for i in xrange(self.leftNodeNum):
	    while(True):
		for j in xrange(self.rightNodeNum):
		    self.t[j] = False
		    self.slack[j] = sys.maxint
		for j in xrange(self.leftNodeNum):
		    self.s[j] = False

		if self.match(i): break	
		else: self.update()
		
	mat = np.array(np.zeros(self.leftNodeNum),dtype=int)
	for i in xrange(self.rightNodeNum):
	    if self.left[i] != -1:
	    	mat[self.left[i]] = i
	return mat
	 

if __name__ == '__main__':
    w = np.array([[0,1,0,2,0],[1,1,0,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,2,0,1,1]])
    k = kuhn_Munkres(w) 
    mat = k.KM()
    sums = 0
    for i in range(len(mat)):
	sums += w[i][mat[i]]
        print mat[i],' ', 
    print ''
    
    print 'total weight = ',sums
