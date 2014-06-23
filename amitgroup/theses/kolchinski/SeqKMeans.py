import numpy as np
import matplotlib.pyplot as plt


class SeqKMeans:
  #trainPts should be a 2D numpy array of first-axis length (# of training points)
  def __init__(self, trainPts):
    self.trainPts = trainPts
    #self.trainMeans()
    #self.trainMeansGrowing()

  #Make new clusters when none is close enough
  #This doesn't work very well - better to focus on mixed number of clusters
  def trainMeansGrowing(self, clusterCutoff = 10.0):
    Ts = self.trainPts
    np.random.shuffle(Ts)
    N = len(Ts)
    self.sideLength = np.sqrt(len(self.trainPts[0]))
    s = self.sideLength
    self.means = [np.copy(Ts[0])]
    self.weights = [1]
    NEW_CLUSTER_CUTOFF = clusterCutoff
 
    for n in range(1, N):
      dists = self.dists_to_means(Ts[n])
      if np.min(dists) < NEW_CLUSTER_CUTOFF:
        closestCluster = np.argmin(dists)
        self.weights[closestCluster] += 1
        delta = (1.0 / self.weights[closestCluster]) * (Ts[n] - self.means[closestCluster])
        self.means[closestCluster] += delta  
      else:
        self.means.append(np.copy(Ts[n]))
        self.weights.append(1)
    
    print "Points per cluster: ", self.weights
    plt.close('all')
    numClusters = len(self.means)
    fig, axes = plt.subplots(1, numClusters, figsize=(0.7*numClusters,1)) 
    for i in range(numClusters):
      curMean = np.copy(self.means[i])
      curAxes = axes[i]
      curAxes.xaxis.set_ticks([])
      curAxes.yaxis.set_ticks([])
      curAxes.imshow(curMean.reshape(s,s), cmap='bone')
    plt.show()

  # Make new clusters when none is close enough - if point likelihood more than
  # 2 standard deviations from mean of cluster, make new cluster with that point
  def trainMeansGrowingBernoulli(self, epsilon = .2, std_cutoff = 3):
    Ts = self.trainPts
    np.random.shuffle(Ts)
    N = len(Ts)
    self.sideLength = np.sqrt(len(self.trainPts[0]))
    s = self.sideLength
    self.means = [np.copy(Ts[0])]
    self.weights = [1]
    self.clusterPoints = [[np.copy(Ts[0])]]
 
    for n in range(1,N):
      probs = self.berProbsPerMean(Ts[n], epsilon)
      bestCluster = np.argmax(probs)
      bestClusterMean = self.means[bestCluster]
      bestClusterPts = self.clusterPoints[bestCluster]
      bestClusterLiks = [self.bernoulliProb(bestClusterMean, v, epsilon) for v in bestClusterPts]
      likMean = np.mean(bestClusterLiks)
      likStdev = np.std(bestClusterLiks)
      
      pointLik = self.bernoulliProb(bestClusterMean, Ts[n], epsilon)

      # If point likelihood is within 6 standard deviations for cluster, or 
      std = likMean / 2.0 if likStdev == 0 else likStdev
      if np.abs(pointLik - likMean) / std < std_cutoff:
        bestClusterPts.append(np.copy(Ts[n]))
        self.weights[bestCluster] += 1
        self.means[bestCluster] = np.mean(bestClusterPts, axis=0)
      else:
        self.means.append(np.copy(Ts[n]))
        self.clusterPoints.append([np.copy(Ts[n])])
        self.weights.append(1)
    
    print "Points per cluster: ", self.weights
    plt.close('all')
    numClusters = len(self.means)
    fig, axes = plt.subplots(1, numClusters, figsize=(0.7*numClusters,1)) 
    for i in range(numClusters):
      curMean = np.copy(self.means[i])
      curAxes = axes[i]
      curAxes.xaxis.set_ticks([])
      curAxes.yaxis.set_ticks([])
      curAxes.imshow(curMean.reshape(s,s), cmap='bone')
    plt.show()


  def trainMeans(self, numClusters, useBernoulli = False, epsilon = 0.2):
    Ts = self.trainPts
    np.random.shuffle(Ts)
    N = len(Ts)
    K = numClusters
    self.sideLength = np.sqrt(len(self.trainPts[0]))
    s = self.sideLength

    #initialize means to first K points
    self.means = (Ts[:K]).astype(np.float64)

    #randomly initialize means
    #self.means = (np.random.rand(K,s*s) > 0.5) * 1.0
    
    #weights is number of examples previously factored into a cluster mean;
    #this influences how much subsequent examples assigned to that cluster
    #move its mean by
    self.weights = np.zeros(K)

    numGraphSteps = 10
    graphEvery = N / numGraphSteps

    plt.close('all')
    fig, axes = plt.subplots(numGraphSteps, K, figsize=(20,20)) 

    np.random.shuffle(Ts)
    for n in range(N):
      closestCluster = self.classify(Ts[n], useBernoulli, epsilon)
      self.weights[closestCluster] += 1
      
      delta = (1.0 / self.weights[closestCluster]) * (Ts[n] - self.means[closestCluster])
      #delta = (Ts[n] - self.means[closestCluster])

      self.means[closestCluster] += delta  
      if n % graphEvery == 0:
        for i in range(K):
          curMean = np.copy(self.means[i])
          curAxes = axes[n/graphEvery, i]
          curAxes.xaxis.set_ticks([])
          curAxes.yaxis.set_ticks([])
          curAxes.imshow(curMean.reshape(s,s), cmap='bone')

    print "Points per cluster: ", self.weights
    plt.show()


  #Use Euclidian distance or BernoulliProbability
  def classify(self, p, useBernoulli = False, epsilon = 0.2):
    #print self.dists_to_means(p)
    if useBernoulli: return np.argmax(self.berProbsPerMean(p, epsilon))
    else:            return np.argmin(self.dists_to_means(p))

  #Euclidian distances
  def dists_to_means(self,p):
    return np.apply_along_axis(np.linalg.norm,1,self.means-p)

  # Bernoulli probability - probs is vector of probabilities, 
  # v is binary vector. Smoothed by epsilon to avoid zero probabilities
  def bernoulliProb(self, probs, v, epsilon):
    berProbs = np.power(probs, v) * np.power(1.0 - probs, 1 - v) 
    return np.product((berProbs == 0) * epsilon + berProbs)

  # Given a point, return a vector of length (# of clusters), of which each
  # entry corresponds to the Bernoulli likelihood for the point relative to
  # the given cluster mean interpreted as a vector of Bernoulli probabilities
  def berProbsPerMean(self, v, epsilon):
    numsOfPts = np.array(self.weights).reshape((len(self.weights), 1))
    adjustedProbs = (self.means * numsOfPts + 2) / (numsOfPts + 4)
    return np.apply_along_axis(lambda p: self.bernoulliProb(p, v, epsilon), 1, self.means)

  def showClusters(self):
    sideLength = np.sqrt(len(self.trainPts[0]))
    clusterGrids = self.means.reshape(self.numClusters,sideLength,sideLength)
    plt.close('all')
    for i in range(self.numClusters):
      plt.subplot(1,10,i+1)
      plt.gca().axes.get_xaxis().set_ticks([])
      plt.gca().axes.get_yaxis().set_ticks([])
      plt.imshow(clusterGrids[i], cmap='bone')
    plt.show()

  #If given N test points, M classes, expect sequentially N/M test points per class
  #However, for this to work, we have to figure out which cluster is which class
  #Nevertheless, confusion matrix is useful
  def test(self, testPts):
    N = len(testPts)
    K = self.numClusters
    errors = np.zeros(K)
    classifications = np.zeros((K,K))
    for n in range(N):
      trueClass = K * n / N
      classedAs = self.classify(testPts[n])
      if classedAs != trueClass: errors[trueClass] += 1
      classifications[trueClass,classedAs] += 1
    errorRates = 1.0 * errors / (N / K)
    #for i in range(K):
    #  print "Error rate {} for class {}".format(errorRates[i], i)
    #print "Overall error rate: {}".format(errorRates.mean())
    print classifications

#import digit_features as df
#skm1 = SeqKMeans(df.flatPixelTrainData())
#skm1.trainMeansGrowingBernoulli()
