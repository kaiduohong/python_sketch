# -*- coding:utf-8 -*-
#update data:20170108:17:41
import pickle
import json
import h5py
import numpy as np
import random


jfile = file("arg.json")                        
argument = json.load(jfile)  

#读取文件
matFileName = argument["matFileName"]            
data = h5py.File(matFileName) 


print 'begin'
#sketchNum = argument["sketchNum"]
samples = np.array(data['samples']).transpose()
[sampleNum,dataDim] = np.shape(samples)
#多维压缩为一维
labels = np.array(data['label']).ravel()
proportion = argument["proportion"]     
index = range(sampleNum)                         

selectedIndex = random.sample(index,int(np.floor(sampleNum * proportion)))                        

trainSamples = np.matrix(samples[selectedIndex]) 
trainLabels = labels[selectedIndex]              
if argument['debug'] == False:
    testIndex = list(set(index) - set(selectedIndex))          
else:
    testIndex = selectedIndex
testSamples = np.matrix(samples[testIndex])      
testLabels = labels[testIndex]                   
print 'begin dump'

#存入文件fileName中,命名规则是multiPie_比例.dat
data  ={}
data['trainSamples'] = trainSamples              
data['trainLabels'] = trainLabels                
data['testSamples'] = testSamples                
data['testLabels'] = testLabels                    
#debug表示test数据和train数据一致
if argument['debug'] == True:
    print 'this is debug data'
    fileName = 'multiPie_'+str(int(proportion * 100)) + '_debug.dat'
else:
    fileName = 'multiPie_'+str(int(proportion * 100)) + '.dat'


f = open(argument['dataFile'],'wb')        
pickle.dump(data,f)
print 'done'
