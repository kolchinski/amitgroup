import numpy as np
from random import randrange, random
import digit_features as df
import matplotlib.pyplot as plt
import synth_features as sf


class UnsupervisedNeuralNet:
  # C is num digits, N num neurons per class, D num features per example
  # theta is field threshold for neurons (>theta is 1, <theta is 0)
  # delta is training threshold: if we aren't at least delta away from the
  # theta threshold for a training point in the correct direction, consider
  # changing the relevant synapse weights to increase separation: bump the
  # synapse weight in the appropriate direction with probability transP
  theta = 1
  delta = 1 
  transP = 0.05
  initTransP = 0.01
  C = 10

  def __init__(self, numClasses, N = 10, delta = 1):
    # Set the number of neurons per class
    self.N = N
    self.delta = delta
    self.C = numClasses

  # Using our synapse weights, classify x as one of the C classes
  def classify(self, x):
    if len(x) != self.D: raise Exception("Incorrect length of vector")
    # Classify the vector by taking the dot product with each set of synapses,
    # and picking the class that has the most # of neurons with dot product > theta
    # return np.argmax(np.sum(np.dot(self.synapses,x) > self.theta, 1))

    onNeuronCounts = np.sum(np.dot(self.synapses,x) > self.theta, 1)
    print onNeuronCounts
    return np.argmax(onNeuronCounts)

  # x is the vector, d is its class
  def trainOnPoint(self, synapses, x, d, transProb):
    # We can increment a synapse if it's not maxed out, and vice versa
    canIncrement = synapses < 1
    canDecrement = synapses > -1

    # Duplicate the input vector C*N times (once per neuron)
    inputIsOn = np.ones((self.C, self.N, self.D)) * x

    # 1 for every synapse of neurons matching the current training point,
    # 0 otherwise
    sameClass = np.zeros((self.C, self.N, self.D))
    sameClass[d] = np.ones((self.N, self.D))

    # A neuron's field is the dot product of synapses times input vector
    print self.C, self.N
    print synapses.shape
    print np.inner(x,synapses).shape
    fields = np.inner(x, synapses).reshape(self.C, self.N, 1)
    # Must have low field to consider incrementing synapse weights
    lowField = (fields < self.theta + self.delta) 
    lowField = lowField * np.ones((self.C, self.N, self.D))
    # Must have high field to consider decrementing synapse weights
    highField = (fields > self.theta - self.delta) 
    highField = highField * np.ones((self.C, self.N, self.D))

    # Only change synapse weights with probability transProb, assuming
    # the other conditions are met
    rands = np.random.random((self.C, self.N, self.D)) < transProb

    synapsePluses = inputIsOn * sameClass * canIncrement * lowField * rands
    synapseMinuses = inputIsOn * (1 - sameClass) * canDecrement * highField * rands

    # Increment synapses meeting incrementation conditions (see above)
    synapses += synapsePluses
    # Decrement synapses meeting decrementation conditions (see above)
    synapses -= synapseMinuses


  def initializeSynapses(self):
    # Initialize synapse array to correct shape
    self.synapses = np.zeros((self.C, self.N, self.D))
    self.synapses += (np.random.random((self.C, self.N, self.D)) < self.initTransP)
    self.synapses -= (np.random.random((self.C, self.N, self.D)) < self.initTransP)

  def randomizedSynapses(self, C, N, D):
    synapses = np.zeros((C, N, D))
    synapses += (np.random.random((C, N, D)) < self.initTransP)
    synapses -= (np.random.random((C, N, D)) < self.initTransP)
    return synapses

  # for each point, make a new cluster if the existing ones aren't good enough
  # otherwise, join it to the best cluster
  def trainOnSetGrowing(self, trainData):
    np.random.shuffle(trainData)
    self.D = trainData[0].shape[0]
    self.C = 1
    synapses = self.randomizedSynapses(1,self.N, self.D)
    self.trainOnPoint(synapses, trainData[0], 0, self.initTransP)

    for p in trainData[1:]:
      onNeuronCounts = np.sum(np.dot(synapses,p) > self.theta, 1)
      print onNeuronCounts
      if np.max(onNeuronCounts) > self.N / 2:
        bestClass = np.argmax(onNeuronCounts)
        self.trainOnPoint(synapses, p, bestClass, self.transP)
      else:
        synapses = np.vstack((synapses,self.randomizedSynapses(1,self.N,self.D)))
        self.C += 1
        self.trainOnPoint(synapses, p, self.C - 1, self.initTransP)

    return synapses




  def trainOnSet(self, trainData):
    # Keep track of how synapse training progresses
    synapseHistory = []

    # How long are the vectors we're classifying? Must be all the same
    self.D = trainData[0].shape[0]
    self.initializeSynapses()
    synapses = self.synapses
    numTrainPts = len(trainData)

    self.pointsPerCluster = [[] for i in range(self.C)]

    np.random.shuffle(trainData)

    print "Training unsupervised neural net on {} examples".format(numTrainPts)
    for i in range(self.C):
      self.trainOnPoint(self.synapses, trainData[i], i, self.initTransP)
      self.pointsPerCluster[i].append(trainData[i])

    numPtsTrained = self.C
    for p in trainData[self.C:]:
      numPtsTrained += 1
      if (numPtsTrained * 100) % numTrainPts == 0:
        print "{}% complete".format(100.0 * numPtsTrained / numTrainPts)
      bestClass = self.classify(p)
      print bestClass
      self.trainOnPoint(self.synapses, p, bestClass, self.transP)
      self.pointsPerCluster[bestClass].append(p)

    
    print "Neural net training complete!"
    return synapseHistory

  def testOnSet(self, testData):
    synapses = self.synapses
    totalErrors = 0
    totalTestPoints = 0
    for c in range(self.C):
      numErrors = 0
      numTestPoints = len(testData[c])
      totalTestPoints += numTestPoints
      for i in range(numTestPoints):
        if not self.classify(testData[c][i]) == c:
          numErrors += 1
          totalErrors += 1
      print "Character ", c, ": ", numErrors, " errors out of ", \
          numTestPoints, "; ", 100.0*numErrors/numTestPoints, " % error"

    print "Total error: ", 100.0*totalErrors/totalTestPoints, "%"

  # Show a graphical representation of the average neuron weights, per class
  # Assume the vectors correspond to square image data
  def displayClassMeans(self):
    print [len(c) for c in self.pointsPerCluster]
    classMeans = self.synapses.mean(axis=1)
    sideLength = np.sqrt(self.D)
    if self.D != sideLength**2: 
      raise Exception("Vectors must be of length n^2 for some integer n")
    plt.close('all')
    fig, axes = plt.subplots(1, self.C, figsize=(self.C,1)) 
    for i in range(self.C):
      w = classMeans[i].reshape((sideLength,sideLength))
      curAxes = axes[i]
      curAxes.imshow(w,cmap='bone')
      curAxes.xaxis.set_ticks([])
      curAxes.yaxis.set_ticks([])
    plt.show()


# With 10 neurons, 10 classes, performance seems to taper off somewhere between
# 5000 and 10000 iterations
def testUnsupervisedNeuralNet():
  trainData = df.flatPixelTrainData()
  synthTrainData = sf.synthData(500,256,128)
  #testData = df.flatPixelTestData().reshape(10,1000,900)

  net = UnsupervisedNeuralNet(20)
  synapses = net.trainOnSetGrowing(trainData)
  #net.testOnSet(testData)
  net.displayClassMeans()
  return synapses
#testNeuralNet()


def testSingleCluster():
  trainData = df.flatPixelTrainData()
  unn = UnsupervisedNeuralNet(20)
  np.random.shuffle(trainData)
  unn.C = 1
  unn.D = 900
  unn.initializeSynapses()
  unn.trainOnPoint(unn.synapses, trainData[0], 0, 0.01)
  for p in trainData:
    unn.classify(p)
