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
#matFileName = u'./data/multiPie.mat'

threadhold = argument["threadhold"]
sketchNum = argument["sketchNum"]
'''
matFileName = argument["matFileName"]
data = h5py.File(matFileName)
#data = sio.loadmat(matFileName)
print np.shape(data['samples'])
samples = np.array(data['samples']).transpose()
[sampleNum,dataDim] = np.shape(samples)
labels = np.array(data['label']).ravel()
 
#samples = np.insert(samples, dataDim, values= 1, axis=1)
''' 
print 'loading'
dataFile = open(argument['dataFile'],'rb')
data = pickle.load(dataFile)
dataFile.close()
print data.keys()
trainSamples = data['trainSamples']
trainLabels = data['trainLabels']
testSamples = data['testSamples']
testLabels = data['testLabels']
if np.min(trainLabels) != 0:
    trainLabels -= np.min(trainLabels)
    testLabels -= np.min(trainLabels)


print 'generate data begin!!!'

lg.info('generate data begin!!!')
'''
 
data = {}
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
data['trainSamples'] = trainSamples
data['trainLabels'] = trainLabels
data['testSamples'] = testSamples
data['testLabe'] = testLabels

f = open('../data/data.dat','w')
pickle.dump(data,f)
dd = pickle.load(f)
disp('done')
 
del index,samples,labels,data,selectedIndex,testIndex
'''
 
time1 = time.time()
[Pieces, numberOfPieces, index] = sketchTrial(trainSamples, sketchNum,threadhold)
psketchTrain = getSketch(trainSamples, Pieces, numberOfPieces, sketchNum)
psketchTest = getSketch(testSamples, Pieces, numberOfPieces, sketchNum)
sketchTime = time.time() - time1
lg.info('genetate sketch time = '+str(sketchTime))
 
 
time1 = time.time()
sketchM = simpleSketch(trainSamples, sketchNum)
originalSketchData = trainSamples * sketchM
print 'dn = ',norm(psketchTrain - originalSketchData,'fro')
originalSketchTestData = testSamples * sketchM
sketchT = time.time() - time1
lg.info('genetate original sketch time = '+str(sketchT))

time1 = time.time()
aff,affSketchM = simpleSketchToAffineSpace(trainSamples,sketchNum)
affSketchTrain = (trainSamples - aff) * affSketchM
affSketchTest = (testSamples - aff) * affSketchM
sketchToAffTime = time.time() - time1
lg.info('generate affsketch time = '+ str(sketchToAffTime))


print 'norm1111 = ',norm(sketchM*sketchM.transpose() - Pieces['rightSubspace'][0].transpose()*Pieces['rightSubspace'][0],'fro') 
print 'norm = ',norm(sketchM.transpose()*sketchM - Pieces['rightSubspace'][0]*Pieces['rightSubspace'][0].transpose(),'fro')

time1 = time.time()
randSketch = randSimpleSketch(trainSamples, sketchNum)
randSketchTrainData = trainSamples * randSketch
randSketchTestData = testSamples * randSketch
randSketchTime = time.time() - time1
lg.info('generate randSketch data time = ' + str(randSketchTime))

time1 = time.time()
[Null, Null, pcaM] = np.linalg.svd(trainSamples, full_matrices=False)
pcaM = np.matrix(pcaM[0:sketchNum].transpose())
pcaTrain = trainSamples * pcaM
pcaTest = testSamples * pcaM
pcaTime = time.time() - time1

time1 = time.time()
dataMean = np.mean(trainSamples,0)
[Null,Null,centeredPcaM] = np.linalg.svd(trainSamples - dataMean,full_matrices=False)
centeredPcaM = np.matrix(centeredPcaM[:sketchNum].transpose())
centeredPcaTrain = (trainSamples - dataMean) * centeredPcaM
centeredPcaTest = (testSamples - dataMean) * centeredPcaM
centeredPcaTime = time.time() - time1


lg.info('dataPca time = '+str(pcaTime))
os.system('clear')
 
print 'begin to classify'
lg.info('begin to classify!')
time1 = time.time()
modelRandSketch = svm.SVC(kernel='linear',C=100.)
modelRandSketch.fit(randSketchTrainData,trainLabels)
res = modelRandSketch.predict(randSketchTestData)
randSketchAccuracy = sum(res == testLabels) / float(len(testLabels))
randSketchTime = time.time() - time1
print 'randSketch done~--------------'
lg.info('randSketch done!----------------')

time1 = time.time()
modelAffSketch = svm.SVC(kernel='linear',C=100.)
modelAffSketch.fit(affSketchTrain,trainLabels) 
res = modelAffSketch.predict(affSketchTest)
affSketchAccuracy = sum(res == testLabels) / float(len(testLabels))
AffSketchTime = time.time() - time1

time1 = time.time()
modelpSketch = svm.SVC(kernel='linear',C=100.)
modelpSketch.fit(psketchTrain,trainLabels)
res = modelpSketch.predict(psketchTest)
pAccuracy = sum(res == testLabels) / float(len(testLabels))
 
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
model = svm.SVC(kernel='linear',C=100.)
model.fit(pcaTrain,trainLabels)
res = model.predict(pcaTest)
pcaAccuracy = sum(res == testLabels) / float(len(testLabels))
print 'pcaSketch done!----------------------'
lg.info('pcaSketch done!----------------------')
pcaTime = time.time() - time1

time1 = time.time()
model = svm.SVC(kernel='linear',C=100.)
model.fit(centeredPcaTrain,trainLabels)
res = model.predict(centeredPcaTest)
centeredPcaAccuracy = sum(res == testLabels) / float(len(testLabels))
print 'centeredPca done!--------------------'
lg.info('centeredPca done!--------------------')
centeredPcaTime = time.time() - time1
 
 
time1 = time.time()
model = svm.SVC(kernel='linear',C=100.)
model.fit(originalSketchData,trainLabels)
res = model.predict(originalSketchTestData)
sketchAccuracy = sum(res == testLabels) / float(len(testLabels))
print 'sketch done!--------------'
lg.info('sketch done!--------------------')
sketchTime = time.time() - time1

lg.info('begin cluster')
print 'begin cluster'
time1 = time.time()
classes = spaceClustering.clustering(trainSamples,sketchNum,threadhold)
acc,nmi_cluster = calNMIACC.calNMIACC(classes,trainLabels)
clusterTime = time.time() - time1

lg.info('cluster acc = '+ str(acc)+'   time= '+ str(clusterTime))
lg.info(' ') 
 

lg.info('numberOfSubspaces = '+ str(numberOfPieces))
lg.info('pacc = '+ str(pAccuracy)+ '    time = '+ str(psketchTime))
lg.info('acc = '+ str(oAccuracy)+ '    time = '+ str(otimes))
lg.info('pcaAcc = '+ str(pcaAccuracy)+ '    time = '+ str(pcaTime))
lg.info('sketchAcc = '+ str(sketchAccuracy)+ ' time  = '+ str(sketchTime))
lg.info('randSketchAcc = '+str(randSketchAccuracy)+ ' time = ' + str(randSketchTime))
lg.info('centeredPcaAcc= '+str(centeredPcaAccuracy) + ' time = ' + str(centeredPcaTime))
lg.info('affSketchAcc= ' + str(affSketchAccuracy) + ' time = '+ str(affSketchTime))
