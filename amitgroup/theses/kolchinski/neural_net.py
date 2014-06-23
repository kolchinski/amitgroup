import numpy as np
from random import randrange, random
import digit_features as df
import matplotlib.pyplot as plt


class NeuralNet:
  # C is num digits, N num neurons per class, D num features per example
  # theta is field threshold for neurons (>theta is 1, <theta is 0)
  # delta is training threshold: if we aren't at least delta away from the
  # theta threshold for a training point in the correct direction, consider
  # changing the relevant synapse weights to increase separation: bump the
  # synapse weight in the appropriate direction with probability transP
  theta = 0
  delta = 1 
  transP = 0.01
  C = 10

  def __init__(self, N = 10, delta = 1):
    # Set the number of neurons per class
    self.N = N
    self.delta = delta

  # Using our synapse weights, classify x as one of the C classes
  def classify(self, x):
    if len(x) != self.D: raise Exception("Incorrect length of vector")
    # Classify the vector by taking the dot product with each set of synapses,
    # and picking the class that has the most # of neurons with dot product > theta
    #return np.argmax(np.sum(np.dot(self.synapses,x) > self.theta, 1))

    neuronsOnPerClass = np.sum(np.dot(self.synapses,x) > self.theta, 1)
    bestClass = np.argmax(neuronsOnPerClass)
    #print neuronsOnPerClass, bestClass
    return bestClass
    # reversedMax = np.argmax(neuronsOnPerClass[::-1])
    # return len(neuronsOnPerClass) - 1 - reversedMax

  #trainData must be "labeled" - see digit_features.py
  def trainOnSet(self, trainData, numIterations):
    # Keep track of how synapse training progresses
    synapseHistory = []

    # How long are the vectors we're classifying? Must be all the same
    self.D = trainData[0][0].shape[0]
    # How many different classes are there? Pull it out of the training data
    # Training data must be of form [(V1, c1), (V2, c2), ...]
    # where Vn is the n'th training vector, and cn is the corresponding class
    # label - every 'c' value is just an integer
    self.C = np.unique(trainData[...,1]).size

    # Initialize synapse array to correct shape
    self.synapses = np.zeros((self.C, self.N, self.D))

    synapses = self.synapses
    T = numIterations
    numTrainPts = len(trainData)

    print "Training neural net on {} examples".format(numIterations)
    for t in range(T):
      if t % (numIterations / 5) == 0:
        print "{}% done".format(100 * t / numIterations)
      # At every sweep through the training points, shuffle the order
      if t % numTrainPts == 0: np.random.shuffle(trainData)
      # Keep track of how the synapse weights change every 10 iterations
      if t % 10 == 0: synapseHistory.append(np.copy(self.synapses))
      (x,d) = trainData[t % numTrainPts]
      #print x,d

      # We can increment a synapse if it's not maxed out, and vice versa
      canIncrement = self.synapses < 1
      canDecrement = self.synapses > -1

      # Duplicate the input vector C*N times (once per neuron)
      inputIsOn = np.ones((self.C, self.N, self.D)) * x

      # 1 for every synapse of neurons matching the current training point,
      # 0 otherwise
      sameClass = np.zeros((self.C, self.N, self.D))
      sameClass[d] = np.ones((self.N, self.D))

      # A neuron's field is the dot product of synapses times input vector
      fields = np.inner(x, synapses).reshape(self.C, self.N, 1)
      # Must have low field to consider incrementing synapse weights
      lowField = (fields < self.theta + self.delta) 
      lowField = lowField * np.ones((self.C, self.N, self.D))
      # Must have high field to consider decrementing synapse weights
      highField = (fields > self.theta - self.delta) 
      highField = highField * np.ones((self.C, self.N, self.D))

      # Only change synapse weights with probability transP, assuming
      # the other conditions are met
      rands = np.random.random((self.C, self.N, self.D)) < self.transP

      synapsePluses = inputIsOn * sameClass * canIncrement * lowField * rands
      synapseMinuses = inputIsOn * (1 - sameClass) * canDecrement * highField * rands

      # Increment synapses meeting incrementation conditions (see above)
      synapses += synapsePluses
      # Decrement synapses meeting decrementation conditions (see above)
      synapses -= synapseMinuses
    
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
def testNeuralNet(numIterations):
  trainData = df.labeled(df.flatPixelTrainData())
  testData = df.flatPixelTestData().reshape(10,1000,900)

  net = NeuralNet(20)
  net.trainOnSet(trainData, numIterations)
  net.testOnSet(testData)
  net.displayClassMeans()
  return net
#testNeuralNet(5000)


