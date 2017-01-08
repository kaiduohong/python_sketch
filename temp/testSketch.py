#-*-coding:utf-8 -*-
import time
import os
from sklearn import svm
from simpleSketch import *
from sketchTrial import  *
from getSketch import *
#import scipy.io as sio
from simpleSketchWithRandom import *
#from belongWhichPieces import *
from scipy.linalg.misc import norm
import numpy.linalg
import h5py
import json
import pickle
import calNMIACC
import spaceClustering
from simpleSketchToAffineSpace import simpleSketchToAffineSpace
import getPCA

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
 
time1 = time.time()
sketchM = simpleSketch(trainSamples, sketchNum)
[originalSketchData,originalSketchTestData] = getSimpleSketchData(trainSamples,testSamples,sketchM)
sketchT = time.time() - time1
lg.info('genetate original sketch time = '+str(sketchT))

time1 = time.time()
aff,affSketchM = simpleSketchToAffineSpace(trainSamples,sketchNum)
affSketchTrain = (trainSamples - aff) * affSketchM
affSketchTest = (testSamples - aff) * affSketchM
affSketchTime = time.time() - time1
lg.info('generate affsketch time = '+ str(affSketchTime))


time1 = time.time()
randSketch = randSimpleSketch(trainSamples, sketchNum)
randSketchTrainData = trainSamples * randSketch
randSketchTestData = testSamples * randSketch
randSketchTime = time.time() - time1
lg.info('generate randSketch data time = ' + str(randSketchTime))

time1 = time.time()
[pcaM,pcaTrain,pcaTest] = getPCAData(trainSamples,testSanokes,sketchNum)
pcaTime = time.time() - time1

time1 = time.time()
[centeredPcaM,centeredPcaTrain,centeredPcaTest] = getPCAData(trainSamples,testSamples,dim,True)
centeredPcaTime = time.time() - time1

lg.info('dataPca time = '+str(pcaTime))
os.system('clear')
 
print 'begin to classify'
lg.info('begin to classify!')
time1 = time.time()
randSjetchAccuracy = svmTest(randSketchTrainData,trainLabels,randSketchTestData,testLabels,'linear',100.)
randSketchTime = time.time() - time1
#should be removed
print randSketchAccuracy
lg.info(randSketchAccuracy)
print 'randSketch done~--------------'
lg.info('randSketch done!----------------')


time1 = time.time()
affSketchAccuracy = svmTest(affSketchTrain,trainLabels,affSketchTest,testLabels,'linear',100.)
AffSketchTime = time.time() - time1

time1 = time.time()
pAccuracy = svmTest(psketchTrain,trainLabels,psketchTest,testLabels,'linear',100.)
 
print 'mutiSubspace sketch done!-----------'
lg.info('pskech done!----------------------')
psketchTime = time.time() - time1
 
 
time1 = time.time()
#model = svm.SVC(kernel='linear',C=100.)
#model.fit(trainSamples,trainLabels)
#res = model.predict(testSamples)
#oAccuracy = sum(res == testLabels) / float(len(testLabels))
oAccuracy = 0
# model = libsvmtrain(trainLabels, trainSamples)
# [~, oAccuracy, ~] = libsvmpredict(testLabels, testSamples, model)
otimes = time.time() - time1
print 'o done---------------------------'
lg.info('o done!----------------------------')
 
time1 = time.time()
pcaAccuracy = svmTest(pcaTrain,trainLabels,pcaTest,testLabels,'linear',100.)
print 'pcaSketch done!----------------------'
lg.info('pcaSketch done!----------------------')
pcaTime = time.time() - time1

time1 = time.time()
centeredPcaAccuracy = svmTest(centeredPcaTrain,trainLabels,centeredPcaTest,testLabels)
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
lg.info('centeredPcaAcc= '+str(centeredPcaAccuracy) + ' time = ' + str(centeredPcaTime))
lg.info('affSketchAcc= ' + str(affSketchAccuracy) + ' time = '+ str(affSketchTime))
