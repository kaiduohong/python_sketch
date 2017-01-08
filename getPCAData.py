#-*-coding:utf-8-*-

def getPCAData(trainSamples,testSamples,dim,centered = False):
    dataMean = .0
    if centered == True:
        dataMean = np.mean(trainSamples,0)

    [Null,Null,centeredPcaM] = np.linalg.svd(trainSamples - dataMean,full_matrices=False)
    centeredPcaM = np.matrix(centeredPcaM[:sketchNum].transpose())
    centeredPcaTrain = (trainSamples - dataMean) * centeredPcaM
    centeredPcaTest = (testSamples - dataMean) * centeredPcaM
    print 'datamean = ',dataMean
    return centeredPcaM,pcaTrain,pcaTest
