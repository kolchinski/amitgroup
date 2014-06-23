import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt

numTrain = 100  #Number of training points per class
numTest = 1000  #Number of test points per class
P_HIGH = .4
P_LOW = .1
P_BGD = .2 #background pixel probability

def posVector(N = 100, M = 10, highP = P_HIGH, bgdP = P_BGD, lowP = P_LOW):
  highPart = random(M) < highP
  lowPart = random(M) < lowP
  bgPart = random(N - 2*M) < bgdP
  return 1 * np.append(np.append(highPart, lowPart), bgPart)

def highOnlyPosVector(N = 100, M = 10, highP = P_HIGH, bgdP = P_BGD):
  highPart = random(M) < highP
  bgPart = random(N-M) < bgdP
  return 1 * np.append(highPart,bgPart)

def negVector(N = 100, M = 10, bgdP = P_BGD):
  return 1 * (random(N) < bgdP)

# N is length of vectors, M is number of high/low (as opposed to background)
# probability pixels
def synthData(ptsPerClass, N = 100, M = 10, highOnly = False, 
    highP = P_HIGH, bgdP = P_BGD, lowP = P_LOW):
  data = np.zeros((2 * ptsPerClass, N))
  for i in range(ptsPerClass):
    data[i] = negVector(N, M, bgdP)
    posVec = highOnlyPosVector(N, M, highP, bgdP) if highOnly \
        else posVector(N, M, highP, bgdP, lowP)
    data[ptsPerClass + i] = posVec
  return data
    
# Decide whether a given vector is positive (1) or negative (0) class,
# assuming we know it was generated using this library
# N is number of high-probability indices = number of low-prob indices
def classifyML(v, N):
  posVectorLik = np.product(np.power(P_HIGH, v[:N])) * \
                 np.product(np.power(1 - P_HIGH, 1 - v[:N])) * \
                 np.product(np.power(P_LOW, v[N:2*N])) * \
                 np.product(np.power(1 - P_LOW, 1 - v[N:2*N]))
  negVectorLik = np.product(np.power(P_BGD, v[:2*N])) * \
                 np.product(np.power(1 - P_BGD, 1 - v[:2*N]))
  return 1 if posVectorLik > negVectorLik else 0


def labeled(data):
  labeledData = []
  L = len(data)
  for i in range(L):
    labeledData.append((data[i],i*2/L))
  return np.array(labeledData)


#older version of function, still here for legacy compatibility
def plotSynthDataSynapseHistory(synapseHistory):
  posSynapseOnes = [(synapseSet[1] == 1).sum(axis=0)[:100].sum()/10000.0 
      for synapseSet in synapseHistory]
  posSynapseNegs = [(synapseSet[1] == -1).sum(axis=0)[:100].sum()/10000.0 
      for synapseSet in synapseHistory]
  negSynapseOnes = [(synapseSet[1] == 1).sum(axis=0)[100:200].sum()/10000.0 
      for synapseSet in synapseHistory]
  negSynapseNegs = [(synapseSet[1] == -1).sum(axis=0)[100:200].sum()/10000.0 
      for synapseSet in synapseHistory]
  plt.figure(figsize=(12,10))
  plt.plot(posSynapseOnes, label="Proportion weight 1 synapses for high-prob region")
  plt.plot(posSynapseNegs, label="Proportion weight -1 synapses for high-prob region")
  plt.plot(negSynapseOnes, label="Proportion weight 1 synapses for low-prob region")
  plt.plot(negSynapseNegs, label="Proportion weight -1 synapses for low-prob region")
  plt.legend(loc=4)
  plt.show()

#only one plot - averaged synapses
def plotSynapseHistory2Class1Plot(synapseHistory, M, N):
  numPerceptronsPerClass = synapseHistory[0].shape[1]
  Ts = np.arange(len(synapseHistory)) * 10
  denom = 1.0 * numPerceptronsPerClass * N
  posRegionPosSynapses = [(synapseSet[1] == 1).sum(axis=0)[:N].sum()/denom 
      for synapseSet in synapseHistory]
  posRegionNegSynapses = [(synapseSet[1] == -1).sum(axis=0)[:N].sum()/denom 
      for synapseSet in synapseHistory]
  posRegionZeroSynapses = [(synapseSet[1] == 0).sum(axis=0)[:N].sum()/denom 
      for synapseSet in synapseHistory]

  posRegionAvgSynapses = np.array(posRegionPosSynapses) - np.array(posRegionNegSynapses)

  denom = 1.0 * numPerceptronsPerClass * (M-N)
  bgdRegionPosSynapses = [(synapseSet[1] == 1).sum(axis=0)[N:].sum()/denom 
      for synapseSet in synapseHistory]
  bgdRegionNegSynapses = [(synapseSet[1] == -1).sum(axis=0)[N:].sum()/denom 
      for synapseSet in synapseHistory]
  bgdRegionZeroSynapses = [(synapseSet[1] == 0).sum(axis=0)[N:].sum()/denom 
      for synapseSet in synapseHistory]

  bgdRegionAvgSynapses = np.array(bgdRegionPosSynapses) - np.array(bgdRegionNegSynapses)

  plt.figure(figsize=(12,10))
  plt.title("Convergence of averaged 3MPC in region")
  plt.xlabel("Iteration number")
  plt.plot(Ts,posRegionAvgSynapses, 'g-', 
      label="Positive class: average synapse value over positive region")
  plt.plot(Ts,bgdRegionAvgSynapses, 'b-', 
      label="Positive class: average synapse value over background region")
  plt.legend(loc=0)
  plt.show()


#newer version of function: still only for two-class synth data
#plots positive-class convergence
def plotSynapseHistoryTwoClass(synapseHistory, M, N, plotSecondClass = False):
  numPerceptronsPerClass = synapseHistory[0].shape[1]
  denom = 1.0 * numPerceptronsPerClass * N
  posRegionPosSynapses = [(synapseSet[1] == 1).sum(axis=0)[:N].sum()/denom 
      for synapseSet in synapseHistory]
  posRegionNegSynapses = [(synapseSet[1] == -1).sum(axis=0)[:N].sum()/denom 
      for synapseSet in synapseHistory]
  posRegionZeroSynapses = [(synapseSet[1] == 0).sum(axis=0)[:N].sum()/denom 
      for synapseSet in synapseHistory]


  #negSynapseOnes = [(synapseSet[1] == 1).sum(axis=0)[100:200].sum()/10000.0 
  #    for synapseSet in synapseHistory]
  #negSynapseNegs = [(synapseSet[1] == -1).sum(axis=0)[100:200].sum()/10000.0 
  #    for synapseSet in synapseHistory]
  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in high-prob region")
  plt.xlabel("Iteration number")
  Ts = np.arange(len(synapseHistory)) * 10
  plt.plot(Ts,posRegionPosSynapses, 'g-',
      label="Positive class: Proportion weight 1 synapses for high-prob region")
  plt.plot(Ts,posRegionNegSynapses, 'r-',
      label="Positive class: Proportion weight -1 synapses for high-prob region")
  plt.plot(Ts,posRegionZeroSynapses, 'b-',
      label="Positive class: Proportion weight 0 synapses for high-prob region")
  #plt.legend(loc=4)
  #plt.show()


  if plotSecondClass:
    posRegionPosSynapses = [(synapseSet[0] == 1).sum(axis=0)[:N].sum()/denom 
        for synapseSet in synapseHistory]
    posRegionNegSynapses = [(synapseSet[0] == -1).sum(axis=0)[:N].sum()/denom 
        for synapseSet in synapseHistory]
    posRegionZeroSynapses = [(synapseSet[0] == 0).sum(axis=0)[:N].sum()/denom 
        for synapseSet in synapseHistory]
    #plt.figure(figsize=(12,10))
    #plt.title("Convergence of negative-class synapses")
    Ts = np.arange(len(synapseHistory)) * 10
    plt.plot(Ts,posRegionPosSynapses, 'g--', 
        label="Negative class: Proportion weight 1 synapses for high-prob region")
    plt.plot(Ts,posRegionNegSynapses, 'r--', 
        label="Negative class: Proportion weight -1 synapses for high-prob region")
    plt.plot(Ts,posRegionZeroSynapses, 'b--', 
        label="Negative class: Proportion weight 0 synapses for high-prob region")

  plt.legend(loc=0)
  plt.show()


  denom = 1.0 * numPerceptronsPerClass * (M-N)
  bgdRegionPosSynapses = [(synapseSet[1] == 1).sum(axis=0)[N:].sum()/denom 
      for synapseSet in synapseHistory]
  bgdRegionNegSynapses = [(synapseSet[1] == -1).sum(axis=0)[N:].sum()/denom 
      for synapseSet in synapseHistory]
  bgdRegionZeroSynapses = [(synapseSet[1] == 0).sum(axis=0)[N:].sum()/denom 
      for synapseSet in synapseHistory]
  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in background region")
  plt.xlabel("Iteration number")
  plt.plot(Ts,bgdRegionPosSynapses, 'g-', 
      label="Positive class: Proportion weight 1 synapses for background region")
  plt.plot(Ts,bgdRegionNegSynapses, 'r-', 
      label="Positive class: Proportion weight -1 synapses for background region")
  plt.plot(Ts,bgdRegionZeroSynapses, 'b-', 
      label="Positive class: Proportion weight 0 synapses for background region")
  #plt.legend(loc=4)
  #plt.show()

  
  if plotSecondClass:
    bgdRegionPosSynapses = [(synapseSet[0] == 1).sum(axis=0)[N:].sum()/denom 
        for synapseSet in synapseHistory]
    bgdRegionNegSynapses = [(synapseSet[0] == -1).sum(axis=0)[N:].sum()/denom 
        for synapseSet in synapseHistory]
    bgdRegionZeroSynapses = [(synapseSet[0] == 0).sum(axis=0)[N:].sum()/denom 
        for synapseSet in synapseHistory]
    #plt.figure(figsize=(12,10))
    #plt.title("Convergence of negative-class synapses")
    plt.plot(Ts,bgdRegionPosSynapses, 'g--', 
        label="Negative class: Proportion weight 1 synapses for background region")
    plt.plot(Ts,bgdRegionNegSynapses, 'r--', 
        label="Negative class: Proportion weight -1 synapses for background region")
    plt.plot(Ts,bgdRegionZeroSynapses, 'b--', 
        label="Negative class: Proportion weight 0 synapses for background region")

  plt.legend(loc=0)
  plt.show()

#newer version of function: still only for two-class synth data
#plots positive-class convergence per perceptron
def plotSynapseHistoriesTwoClass(synapseHistory, M, N):
  numPerceptronsPerClass = synapseHistory[0].shape[1]
  Ts = np.arange(len(synapseHistory)) * 10

  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in high-prob region")
  plt.xlabel("Iteration number")
  denom = 1.0 * N
  
  for i in range(numPerceptronsPerClass):
    posRegionPosSynapses = [(synapseSet[1][i] == 1)[:N].sum()/denom 
        for synapseSet in synapseHistory]
    posRegionNegSynapses = [(synapseSet[1][i] == -1)[:N].sum()/denom 
        for synapseSet in synapseHistory]
    posRegionZeroSynapses = [(synapseSet[1][i] == 0)[:N].sum()/denom 
        for synapseSet in synapseHistory]

    plt.plot(Ts,posRegionPosSynapses, 'g-',
      label="Positive class: Proportion weight 1 synapses for high-prob region" if i == 0 else None)
    plt.plot(Ts,posRegionNegSynapses, 'r-',
      label="Positive class: Proportion weight -1 synapses for high-prob region" if i == 0 else None)
    plt.plot(Ts,posRegionZeroSynapses, 'b-',
      label="Positive class: Proportion weight 0 synapses for high-prob region" if i == 0 else None)

  plt.legend(loc=0)
  plt.show()


  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in background region")
  plt.xlabel("Iteration number")
  denom = 1.0 * (M-N)

  for i in range(numPerceptronsPerClass):
    bgdRegionPosSynapses = [(synapseSet[1][i] == 1)[N:].sum()/denom 
        for synapseSet in synapseHistory]
    bgdRegionNegSynapses = [(synapseSet[1][i] == -1)[N:].sum()/denom 
        for synapseSet in synapseHistory]
    bgdRegionZeroSynapses = [(synapseSet[1][i] == 0)[N:].sum()/denom 
        for synapseSet in synapseHistory]

    plt.plot(Ts,bgdRegionPosSynapses, 'g-', 
      label="Positive class: Proportion weight 1 synapses for background region" if i==0 else None)
    plt.plot(Ts,bgdRegionNegSynapses, 'r-', 
      label="Positive class: Proportion weight -1 synapses for background region" if i==0 else None)
    plt.plot(Ts,bgdRegionZeroSynapses, 'b-', 
      label="Positive class: Proportion weight 0 synapses for background region" if i==0 else None)
  plt.legend(loc=0)
  plt.show()





#newer version of function: still only for two-class synth data
#plots positive-class convergence per perceptron
#this is for lo/hi/bgd, one of two classes
def plotSynapseHistoriesThreeRegion(synapseHistory, M, N):
  numPerceptronsPerClass = synapseHistory[0].shape[1]
  Ts = np.arange(len(synapseHistory)) * 10

  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in high-prob region")
  plt.xlabel("Iteration number")
  denom = 1.0 * N
  
  for i in range(numPerceptronsPerClass):
    posRegionPosSynapses = [(synapseSet[1][i] == 1)[:N].sum()/denom 
        for synapseSet in synapseHistory]
    posRegionNegSynapses = [(synapseSet[1][i] == -1)[:N].sum()/denom 
        for synapseSet in synapseHistory]
    posRegionZeroSynapses = [(synapseSet[1][i] == 0)[:N].sum()/denom 
        for synapseSet in synapseHistory]

    plt.plot(Ts,posRegionPosSynapses, 'g-',
      label="Positive class: Proportion weight 1 synapses for high-prob region" if i == 0 else None)
    plt.plot(Ts,posRegionNegSynapses, 'r-',
      label="Positive class: Proportion weight -1 synapses for high-prob region" if i == 0 else None)
    plt.plot(Ts,posRegionZeroSynapses, 'b-',
      label="Positive class: Proportion weight 0 synapses for high-prob region" if i == 0 else None)

  plt.legend(loc=0)
  plt.show()


  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in low-prob region")
  plt.xlabel("Iteration number")
  denom = 1.0 * N
  
  for i in range(numPerceptronsPerClass):
    lowRegionPosSynapses = [(synapseSet[1][i] == 1)[N:2*N].sum()/denom 
        for synapseSet in synapseHistory]
    lowRegionNegSynapses = [(synapseSet[1][i] == -1)[N:2*N].sum()/denom 
        for synapseSet in synapseHistory]
    lowRegionZeroSynapses = [(synapseSet[1][i] == 0)[N:2*N].sum()/denom 
        for synapseSet in synapseHistory]

    plt.plot(Ts,lowRegionPosSynapses, 'g-',
      label="Positive class: Proportion weight 1 synapses for low-prob region" if i == 0 else None)
    plt.plot(Ts,lowRegionNegSynapses, 'r-',
      label="Positive class: Proportion weight -1 synapses for low-prob region" if i == 0 else None)
    plt.plot(Ts,lowRegionZeroSynapses, 'b-',
      label="Positive class: Proportion weight 0 synapses for low-prob region" if i == 0 else None)

  plt.legend(loc=0)
  plt.show()



  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in background region")
  plt.xlabel("Iteration number")
  denom = 1.0 * (M-2*N)

  for i in range(numPerceptronsPerClass):
    bgdRegionPosSynapses = [(synapseSet[1][i] == 1)[2*N:].sum()/denom 
        for synapseSet in synapseHistory]
    bgdRegionNegSynapses = [(synapseSet[1][i] == -1)[2*N:].sum()/denom 
        for synapseSet in synapseHistory]
    bgdRegionZeroSynapses = [(synapseSet[1][i] == 0)[2*N:].sum()/denom 
        for synapseSet in synapseHistory]

    plt.plot(Ts,bgdRegionPosSynapses, 'g-', 
      label="Positive class: Proportion weight 1 synapses for background region" if i==0 else None)
    plt.plot(Ts,bgdRegionNegSynapses, 'r-', 
      label="Positive class: Proportion weight -1 synapses for background region" if i==0 else None)
    plt.plot(Ts,bgdRegionZeroSynapses, 'b-', 
      label="Positive class: Proportion weight 0 synapses for background region" if i==0 else None)
  plt.legend(loc=0)
  plt.show()


#newer version of function: still only for two-class synth data
#plots positive-class convergence averaged over perceptrons
#this is for lo/hi/bgd, one of two classes
def plotSynapseHistoryThreeRegion(synapseHistory, M, N):
  numPerceptronsPerClass = synapseHistory[0].shape[1]
  Ts = np.arange(len(synapseHistory)) * 10

  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in high-prob region")
  plt.xlabel("Iteration number")
  denom = 1.0 * N * numPerceptronsPerClass
  
  posRegionPosSynapses = [(synapseSet[1] == 1).sum(axis=0)[:N].sum()/denom 
      for synapseSet in synapseHistory]
  posRegionNegSynapses = [(synapseSet[1] == -1).sum(axis=0)[:N].sum()/denom 
      for synapseSet in synapseHistory]
  posRegionZeroSynapses = [(synapseSet[1] == 0).sum(axis=0)[:N].sum()/denom 
      for synapseSet in synapseHistory]

  plt.plot(Ts,posRegionPosSynapses, 'g-',
    label="Positive class: Proportion weight 1 synapses for high-prob region")
  plt.plot(Ts,posRegionNegSynapses, 'r-',
    label="Positive class: Proportion weight -1 synapses for high-prob region")
  plt.plot(Ts,posRegionZeroSynapses, 'b-',
    label="Positive class: Proportion weight 0 synapses for high-prob region")

  plt.legend(loc=0)
  plt.show()


  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in low-prob region")
  plt.xlabel("Iteration number")
  denom = 1.0 * N * numPerceptronsPerClass
  
  lowRegionPosSynapses = [(synapseSet[1] == 1).sum(axis=0)[N:2*N].sum()/denom 
      for synapseSet in synapseHistory]
  lowRegionNegSynapses = [(synapseSet[1] == -1).sum(axis=0)[N:2*N].sum()/denom 
      for synapseSet in synapseHistory]
  lowRegionZeroSynapses = [(synapseSet[1] == 0).sum(axis=0)[N:2*N].sum()/denom 
      for synapseSet in synapseHistory]

  plt.plot(Ts,lowRegionPosSynapses, 'g-',
    label="Positive class: Proportion weight 1 synapses for low-prob region")
  plt.plot(Ts,lowRegionNegSynapses, 'r-',
    label="Positive class: Proportion weight -1 synapses for low-prob region")
  plt.plot(Ts,lowRegionZeroSynapses, 'b-',
    label="Positive class: Proportion weight 0 synapses for low-prob region")

  plt.legend(loc=0)
  plt.show()



  plt.figure(figsize=(12,10))
  plt.title("Convergence of synapses in background region")
  plt.xlabel("Iteration number")
  denom = 1.0 * (M - 2*N) * numPerceptronsPerClass

  bgdRegionPosSynapses = [(synapseSet[1] == 1).sum(axis=0)[2*N:].sum()/denom 
      for synapseSet in synapseHistory]
  bgdRegionNegSynapses = [(synapseSet[1] == -1).sum(axis=0)[2*N:].sum()/denom 
      for synapseSet in synapseHistory]
  bgdRegionZeroSynapses = [(synapseSet[1] == 0).sum(axis=0)[2*N:].sum()/denom 
      for synapseSet in synapseHistory]

  plt.plot(Ts,bgdRegionPosSynapses, 'g-', 
    label="Positive class: Proportion weight 1 synapses for background region")
  plt.plot(Ts,bgdRegionNegSynapses, 'r-', 
    label="Positive class: Proportion weight -1 synapses for background region")
  plt.plot(Ts,bgdRegionZeroSynapses, 'b-', 
    label="Positive class: Proportion weight 0 synapses for background region")
  plt.legend(loc=0)
  plt.show()


#Also write a per-perceptron version
#Also look at low/bgd/hi vs bgd again, with more detailed graphs
#demonstrate that the two classes are effectively inverses and only look at one?








