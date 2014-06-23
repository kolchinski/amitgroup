from sklearn import svm
import numpy as np

# Quadratic time since pairwise SVMs - implemented with libSVM
# classification rows (0th axis) true class, cols (1th axis) classified-as
def oneVsOne(trainData, testData, numClasses):
  classifyWithSVM(trainData, testData, numClasses, svm.SVC)

# Implemented with liblinear
# classification rows (0th axis) true class, cols (1th axis) classified-as
def oneVsRest(trainData, testData, numClasses):
  classifyWithSVM(trainData, testData, numClasses, svm.LinearSVC)

def classifyWithSVM(trainData, testData, numClasses, SVMFn):
  X = trainData
  Y = np.arange(len(trainData)) / (len(trainData) / numClasses)
  clf = SVMFn()
  clf.fit(X,Y)
  predictions = clf.predict(testData)
  testTruth = np.arange(len(testData)) / (len(testData) / numClasses)
  print "Percent error: ", np.sum(testTruth != predictions) / (1.0 * len(testData))
  classifications = np.zeros((numClasses,numClasses))
  for i in range(len(testData)):
      classifications[i/(len(testData)/numClasses)][predictions[i]] += 1
  print(classifications)


# Example:
# import digit_features as df
# import sklearn_digit_svm as skl
# skl.oneVsOne(df.flatEdgeTrainData(), df.flatEdgeTestData(), 10)
# skl.oneVsRest(df.flatEdgeTrainData(), df.flatEdgeTestData(), 10)

