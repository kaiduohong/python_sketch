#-*-coding:utf-8 -*-
import os
os.system('rm *.pyc')

import time
import os
from sklearn import svm
from simpleSketch import *
#from sketchTrial import  *
#from getSketch import *
#import scipy.io as sio
from simpleSketchWithRandom import *
from weightSimpleSketch import *
#from belongWhichPieces import *
from scipy.linalg.misc import norm
import numpy.linalg
import h5py
import json
import pickle
#import calNMIACC
##import spaceClustering
from simpleSketchToAffineSpace import simpleSketchToAffineSpace
from getPCAData import getPCAData
from svmTest import svmTest

'''
reload(getPCAData)
reload(svmTest)
reload(simpleSketch)
reload(simpleSketchWithRandom)
reload(simpleSketchToAffineSpace)
'''

jfile = file("arg.json")
print jfile
argument = json.load(jfile)
jfile.close()

logFile = raw_input('logFileName\n')
lg.basicConfig(level=lg.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                #datefmt='%a,%d,%b,%Y,%H:%M:%S',
                filename='./log/'+logFile,
                filemode='w')

threadhold = argument["threadhold"]
sketchNum = argument["sketchNum"]

print 'loading data'
dataFile = open(argument['dataFile'],'rb')
data = pickle.load(dataFile)
dataFile.close()
trainSamples = data['trainSamples']
trainLabels = data['trainLabels']
testSamples = data['testSamples']
testLabels = data['testLabels']
if np.min(trainLabels) != 0:
    minimum = np.min(trainLabels)
    print 'minimun = ',minimum
    trainLabels -= minimum
    testLabels -= minimum
print 'size of trainLabels = ',np.shape(trainLabels)


print 'generate data begin!!!'
lg.info('generate data begin!!!')

'''

matFileName = u'./data/multiPie.mat'
matFileName = argument["matFileName"]
data = h5py.File(matFileName)
#data = sio.loadmat(matFileName)
print np.shape(data['samples'])
samples = np.array(data['samples']).transpose()
[sampleNum,dataDim] = np.shape(samples)
labels = np.array(data['label']).ravel()
 
#samples = np.insert(samples, dataDim, values= 1, axis=1)
 
proportion = argument["proportion"]
index = range(sampleNum)
selectedIndex = random.sample(index,int(np.floor(sampleNum * proportion)))
trainSamples = np.matrix(samples[selectedIndex])
trainLabels = labels[selectedIndex]
if argument["trainErr"]:
    testIndex = selectedIndex
else: 
    testIndex = list(set(index) - set(selectedIndex))
 
testSamples = np.matrix(samples[testIndex])
testLabels = labels[testIndex]

 
del index,samples,labels,selectedIndex,testIndex
'''

#进行降维,生成数据
#生成子空间，先注释掉，现在没啥用
'''
time1 = time.time()
[Pieces, numberOfPieces, index] = sketchTrial(trainSamples, sketchNum,threadhold)
psketchTrain = getSketch(trainSamples, Pieces, numberOfPieces, sketchNum)
psketchTest = getSketch(testSamples, Pieces, numberOfPieces, sketchNum)
sketchTime = time.time() - time1
lg.info('genetate sketch time = '+str(sketchTime))
'''

#原来的sketch方法 
time1 = time.time()
sketchM = simpleSketch(trainSamples, sketchNum)
[originalSketchData,originalSketchTestData] = getSimpleSketchData(trainSamples,testSamples,sketchM)
sketchT = time.time() - time1
lg.info('genetate original sketch time = '+str(sketchT))

#加权sketch
time1 = time.time()
sketchM = weightSimpleSketch(trainSamples, sketchNum)
[weightSketchTrain,weightSketchTest] = getWeightSimpleSketchData(trainSamples,testSamples,sketchM)
weightSketchT = time.time() - time1
lg.info('generate weightSketch time = '+str(weightSketchT))

#带平移中心化的sketch
time1 = time.time()
aff,affSketchM = simpleSketchToAffineSpace(trainSamples,sketchNum)
affSketchTrain = (trainSamples - aff) * affSketchM
affSketchTest = (testSamples - aff) * affSketchM
affSketchTime = time.time() - time1
lg.info('generate affsketch time = '+ str(affSketchTime))


#rand Sketch
time1 = time.time()
randSketch = randSimpleSketch(trainSamples, sketchNum)
randSketchTrainData = trainSamples * randSketch
randSketchTestData = testSamples * randSketch
randSketchTime = time.time() - time1
lg.info('generate randSketch data time = ' + str(randSketchTime))

#pca
time1 = time.time()
[pcaM,pcaTrain,pcaTest] = getPCAData(trainSamples,testSamples,sketchNum)
pcaTime = time.time() - time1

#centered PCA
time1 = time.time()
[centeredPcaM,centeredPcaTrain,centeredPcaTest] = getPCAData(trainSamples,testSamples,sketchNum,True)
centeredPcaTime = time.time() - time1

lg.info('dataPca time = '+str(pcaTime))
os.system('clear')
 
print 'begin to classify'
lg.info('begin to classify!')
time1 = time.time()
randSketchAccuracy = svmTest(randSketchTrainData,trainLabels,randSketchTestData,testLabels,'linear',100.)
randSketchTime = time.time() - time1
#should be removed
print randSketchAccuracy
lg.info(randSketchAccuracy)
print 'randSketch done~--------------'
lg.info('randSketch done!----------------')


#带平移的sketch
time1 = time.time()
affSketchAccuracy = svmTest(affSketchTrain,trainLabels,affSketchTest,testLabels,'linear',100.)
AffSketchTime = time.time() - time1

'''
time1 = time.time()
pAccuracy = svmTest(psketchTrain,trainLabels,psketchTest,testLabels,'linear',100.)
 
print 'mutiSubspace sketch done!-----------'
lg.info('pskech done!----------------------')
psketchTime = time.time() - time1
'''
 
time1 = time.time()
oAccuracy = svmTest(trainSamples,trainLabels,testSamples,testLabels,'linear',100.)
#oAccuracy = 0
otimes = time.time() - time1
print 'o done---------------------------'
lg.info('o done!----------------------------')

time1 = time.time()
weightSketchAccuracy = svmTest(trainSamples, trainLabels, testSamples,testLabels,'linear',100.)
wtimes = time.times() - time1
print 'weightSketch done!-------------------------'
lg.info('weightSketch done!-------------------------')
 
time1 = time.time()
pcaAccuracy = svmTest(pcaTrain,trainLabels,pcaTest,testLabels,'linear',100.)
print 'pcaSketch done!----------------------'
lg.info('pcaSketch done!----------------------')
pcaTime = time.time() - time1

time1 = time.time()
centeredPcaAccuracy = svmTest(centeredPcaTrain,trainLabels,centeredPcaTest,testLabels,'linear',100.)
print 'centeredPca done!--------------------'
lg.info('centeredPca done!--------------------')
centeredPcaTime = time.time() - time1
 
 
time1 = time.time()
sketchAccuracy = svmTest(originalSketchData,trainLabels,originalSketchTestData,testLabels,'linear',100.)
print 'sketch done!--------------'
lg.info('sketch done!--------------------')
sketchTime = time.time() - time1

'''
lg.info('begin cluster')
print 'begin cluster'
time1 = time.time()
classes = spaceClustering.clustering(trainSamples,sketchNum,threadhold)
acc,nmi_cluster = calNMIACC.calNMIACC(classes,trainLabels)
clusterTime = time.time() - time1

lg.info('cluster acc = '+ str(acc)+'   time= '+ str(clusterTime))
lg.info(classes)
lg.info(' ') 
''' 

#lg.info('numberOfSubspaces = '+ str(numberOfPieces))
#lg.info('pacc = '+ str(pAccuracy)+ '    time = '+ str(psketchTime))
lg.info('acc = '+ str(oAccuracy)+ '    time = '+ str(otimes))
lg.info('pcaAcc = '+ str(pcaAccuracy)+ '    time = '+ str(pcaTime))
lg.info('sketchAcc = '+ str(sketchAccuracy)+ ' time  = '+ str(sketchTime))
lg.info('randSketchAcc = '+str(randSketchAccuracy)+ ' time = ' + str(randSketchTime))
lg.info('weightSketchAcc = '+str(weightSketchAccuracy)+' time = '+str(weightSketchT))
lg.info('centeredPcaAcc= '+str(centeredPcaAccuracy) + ' time = ' + str(centeredPcaTime))
lg.info('affSketchAcc= ' + str(affSketchAccuracy) + ' time = '+ str(affSketchTime))

