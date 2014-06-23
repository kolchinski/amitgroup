import numpy as np
from scipy import misc
from random import randrange, getrandbits, choice,sample
import matplotlib.pyplot as plt
import digit_features as df

class PegasosSVM:
  l = 0.5 # learning rate
  C = 10  # Number of classes

  def __init__(self, numClasses = 10):
    self.C = numClasses 

  def classify(self, x):
    dots = [np.dot(self.W[d], x) for d in range(self.C)]
    return dots.index(max(dots))

  # trainData must be "labeled" - see digit_features.py
  # unlabeledData must be of length 10*N, where N is number of training points
  # per class - this is really not generalizable to arbitrary lengths
  def trainOnSet(self, trainData, unlabeledTrainData, numIterations):
    S = len(trainData)        # number of training points
    T = numIterations         # number of training points to train on
    D = trainData[0][0].size  # dimensions of training vectors
    self.W = np.zeros((self.C, D)) # SVM weights (perpendicular to separating plane)
    W = self.W

    weightHistory = np.zeros((self.C, T / 10, D))

    #Train one-vs-the-rest classifiers one by one
    #This could be vectorized, but it runs quickly as it is
    for c in range(self.C):
      print "\nTraining one-vs-rest SVM for class {}".format(c)
      for t in range(1, T + 1):
        if (t-1) % 10 == 0: weightHistory[c,(t-1)/10] = self.W[c]

        # Every full sweep through the training points, shuffle their order
        if t % S == 0: np.random.shuffle(trainData)

        #Every 20% of training points, print out total loss
        if t % (T/5) == 0: 
          weightPenalty = self.l / 2.0 * np.linalg.norm(W[c])**2
          dotProducts = np.inner(unlabeledTrainData, W[c])
          #Ys is 1 if class matches, -1 if not
          Ys = (np.arange(S) / (S/self.C) == c) * 2 - 1
          losses = (1 - Ys * dotProducts)
          positiveLosses = (losses > 0) * losses
          avgLoss = np.mean(positiveLosses)
          totalLoss = weightPenalty + avgLoss
          print "Total loss: {}".format(totalLoss)

        AsubT = trainData[t%S]
        #etaT = 1.0 / l / (t*5 + 1)
        etaT = 1.0 / self.l / (t + 1)
        W[c] *= 1.0 * (t - 1) / t

        (x,d) = AsubT
        y = 1 if (d == c) else -1
        if np.dot(W[c], x) * y < 1:
          W[c] += etaT * y * x
        
        wNorm = np.linalg.norm(W[c])
        maxNorm = 1.0 / np.sqrt(self.l)
        if wNorm > maxNorm: 
          W[c] *= (maxNorm / wNorm)

    return weightHistory
          
  def testOnSet(self, testData):
    classifications = np.zeros((self.C,self.C))

    totalErrors = 0
    numTestPoints = len(testData)
    for i in range(numTestPoints):
      x,c = testData[i]
      classedAs = self.classify(x)
      classifications[c,classedAs] += 1
      if not classedAs == c:
        totalErrors += 1

    print "Total error: ", 100.0*totalErrors/numTestPoints, "%"
    for c in range(self.C):
      rights = classifications[c][c]
      total = np.sum(classifications[c])
      print "Error for class {}: {}%".format(c, 100.0*(total - rights)/total)
    print classifications

  def displayEdgeWeights(self):
    # graph learned digit representations, hardcoded for edges
    plt.close('all')
    fig, axes = plt.subplots(self.C, 9, figsize=(self.C,9)) 
    meanedges = np.mean(self.W.reshape(10,30,30,8), axis=3)
    for i in range(self.C):
      w = self.W[i].reshape((30,30,8))
      axes[i,0].imshow(meanedges[i], cmap='Reds')
      axes[i,0].xaxis.set_ticks([])
      axes[i,0].yaxis.set_ticks([])
      #plt.subplot(self.C,9,9*i+1)
      #plt.imshow(meanedges[i], cmap='Reds')
      for j in range(8):
        #plt.subplot(self.C,9,9*i+j+2)
        #plt.imshow(w[:,:,j], cmap='bone')
        axes[i,j+1].imshow(w[:,:,j], cmap='bone')
        axes[i,j+1].xaxis.set_ticks([])
        axes[i,j+1].yaxis.set_ticks([])
        
    plt.show()

  def displayPixelWeights(self):
    # graph learned digit representations, hardcoded for pixels 
    plt.close('all')
    fig, axes = plt.subplots(1, self.C, figsize=(self.C,1)) 
    for i in range(self.C):
      w = self.W[i].reshape((30,30))
      curAxes = axes[i]
      curAxes.imshow(w,cmap='bone')
      curAxes.xaxis.set_ticks([])
      curAxes.yaxis.set_ticks([])
    plt.show()

  def showProductHistogram(self, testData):
    # Histogram of digit-weight dot products
    plt.close('all')
    fig, axes = plt.subplots(self.C, self.C, figsize=(1.5*self.C,1.5*self.C)) 
    for wtClass in range(10):
      for imgClass in range(10):
        #plt.subplot(10,10,1+wtClass*10+imgClass)
        products = np.inner(testData[imgClass*1000:(imgClass+1)*1000],self.W[wtClass])
        axes[wtClass,imgClass].hist(products, range=(-5,5))
    plt.tight_layout()
    plt.show()


def testPegasos():
  unlabeledTrainData = df.flatEdgeTrainData()
  unlabeledTestData = df.flatEdgeTestData()
  trainData = df.labeled(unlabeledTrainData)
  testData = df.labeled(unlabeledTestData)
  peg = PegasosSVM()
  peg.trainOnSet(trainData,unlabeledTrainData,8000)
  peg.testOnSet(testData)
  peg.displayWeights()
  peg.showProductHistogram(unlabeledTestData)
  return peg
#peg = testPegasos()

#import digit_features as df
#pixelPegasos = PegasosSVM()
#pixelPegasos.trainOnSet(df.labeled(df.flatPixelTrainData()), df.flatPixelTrainData(), 8000)
#pixelPegasos.displayPixelWeights()
#edgePegasos = PegasosSVM()
#edgePegasos.trainOnSet(df.labeled(df.flatEdgeTrainData()), df.flatEdgeTrainData(), 8000)
#edgePegasos.displayEdgeWeights()
