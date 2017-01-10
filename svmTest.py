#-*-coding:utf-8 -*-
from sklearn import svm
def svmTest(trainSamples,trainLabels,testSamples,testLabels,k,c):
    model = svm.SVC(kernel=k,C=c)
    model.fit(trainSamples,trainLabels)
    res = model.predict(testSamples)
    acc = sum(res == testLabels) / float(len(testLabels))
    return acc
