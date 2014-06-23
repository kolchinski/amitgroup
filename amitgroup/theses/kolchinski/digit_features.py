import numpy as np
import amitgroup as ag

numTrain = 100  #Number of training points per class
numTest = 1000  #Number of test points per class
C = 10          #Number of classes
S = 30          #Number of pixels per side
trainPrefix = 'tr'
testPrefix = 'te'
D = S*S

def labeled(data, numClasses = 10):
  labeledData = []
  L = len(data)
  for i in range(L):
    labeledData.append((data[i],i*numClasses/L))
  return np.array(labeledData)

def pixelData(prefix,N):
  trainData = np.zeros((N*C,S,S))
  for i in range(C):
    with open('../data/{}{}.bin'.format(prefix,i)) as f: 
      for j in range(N):
        image = f.read(D)
        if image == "": break
        imgArray = np.array([ord(c) for c in image]).reshape((30,30)) 
        trainData[i*N + j] = imgArray 
  return trainData

def pixelTrainData():
  return pixelData(trainPrefix, numTrain)

def pixelTestData():
  return pixelData(testPrefix, numTest)

def flatPixelTrainData():
  return pixelTrainData().reshape((numTrain*C,D))

def flatPixelTestData():
  return pixelTestData().reshape((numTest*C,D))

def edgeTrainData():
  return ag.features.bedges(pixelTrainData(), k=5, radius=2)

def edgeTestData():
  return ag.features.bedges(pixelTestData(), k=5, radius=2)

def flatEdgeTrainData():
  return edgeTrainData().reshape((numTrain*C,8*D))

def flatEdgeTestData():
  return edgeTestData().reshape((numTest*C,8*D))
