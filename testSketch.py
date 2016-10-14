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

jfile = file("arg.json")
argument = json.load(jfile)
jfile.close()

logFile = raw_input('logFileName\n')
lg.basicConfig(level=lg.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a,%d,%b,%Y,%H:%M:%S',
                filename='./log/'+logFile,
                filemode='w')
#matFileName = u'./data/multiPie.mat'

threadhold = argument["threadhold"]
'''
matFileName = argument["matFileName"]
data = h5py.File(matFileName)
#data = sio.loadmat(matFileName)
print np.shape(data['samples'])
sketchNum = argument["sketchNum"]
samples = np.array(data['samples']).transpose()
[sampleNum,dataDim] = np.shape(samples)
labels = np.array(data['label']).ravel()
 
#samples = np.insert(samples, dataDim, values= 1, axis=1)
''' 
disp('loading')
dataFile = open(argument,'r')
data = pickle.load(dataFile)
dataFile.close()
trainSamples = data['trainSamples']
trainLabels = data['trainLabels']
testSamples = data['testSamples']
testLabels = data['testLabels']


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
centeredPcaTrain = (trainSample - dataMean) * centeredPcaM
centeredPcaTest = (testSample - dataMean) * centeredPcaM
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
 
 
lg.info('numberOfPieces = '+ str(numberOfPieces))
lg.info('paccuracy = '+ str(pAccuracy)+ '    time = '+ str(psketchTime))
lg.info('accuracy = '+ str(oAccuracy)+ '    time = '+ str(otimes))
lg.info('pcaAccuracy = '+ str(pcaAccuracy)+ '    time = '+ str(pcaTime))
lg.info('sketchAccuracy = '+ str(sketchAccuracy)+ '       time  = '+ str(sketchTime))
lg.info('randSketchAccuracy = '+str(randSketchAccuracy)+ ' time = ' + str(randSketchTime))
lg.info('centeredPcaAccuracy= '+str(centeredPcaAccuracy) + ' time = ' = str(centeredPcaTime))
