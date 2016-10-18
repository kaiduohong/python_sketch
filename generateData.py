import pickle
import json
import h5py
import numpy as np
import random


jfile = file("arg.json")                        
argument = json.load(jfile)  

matFileName = argument["matFileName"]            
data = h5py.File(matFileName) 

print 'begin'
sketchNum = argument["sketchNum"]
samples = np.array(data['samples']).transpose()
[sampleNum,dataDim] = np.shape(samples)
labels = np.array(data['label']).ravel()
proportion = argument["proportion"]     
index = range(sampleNum)                         

selectedIndex = random.sample(index,int(np.floor(sampleNum * proportion)))                        

trainSamples = np.matrix(samples[selectedIndex]) 
trainLabels = labels[selectedIndex]              
testIndex = list(set(index) - set(selectedIndex))          
testSamples = np.matrix(samples[testIndex])      
testLabels = labels[testIndex]                   
print 'begin dump'

data  ={}
data['trainSamples'] = trainSamples              
data['trainLabels'] = trainLabels                
data['testSamples'] = testSamples                
data['testLabe'] = testLabels                    
fileName = 'multiPie_'+str(int(proportion * 100)) + '.dat'
f = open('../data/data.dat','wb')        
pickle.dump(data,f)
print 'loading'
dd = pickle.load(f)
print dd.keys()
print 'done'
