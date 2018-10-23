

import os
import numpy as np



class KNNClassifier():
    """This is a Nearest Neighbor classifier. """

    def __init__(self, k=3):
        self._k = k

    def _calEDistance(self, inSample, dataset):
        m = dataset.shape[0]
        diffMat = np.tile(inSample, (m, 1)) - dataset
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        return distances.argsort()

    def _classify0(self, inX, dataSet, labels):
        k = self._k
        dataSetSize = dataSet.shape[0]
        diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def _classify(self, sample, train_X, train_y):

        if isinstance(sample, np.ndarray) and isinstance(train_X, np.ndarray) \
                and isinstance(train_y, np.ndarray):
            pass
        else:
            try:
                sample = np.array(sample)
                train_X = np.array(train_X)
                train_y = np.array(train_y)
            except:
                raise TypeError("numpy.ndarray required for train_X and ..")
        sortedDistances = self._calEDistance(sample, train_X)
        classCount = {}
        for i in range(self._k):
            oneVote = train_y[sortedDistances[i]]
            classCount[oneVote] = classCount.get(oneVote, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

        return sortedClassCount[0][0]

    def classify(self, test_X, train_X, train_y):
        results = []

        if isinstance(test_X, np.ndarray) and isinstance(train_X, np.ndarray) \
                and isinstance(train_y, np.ndarray):
            pass
        else:
            try:
                test_X = np.array(test_X)
                train_X = np.array(train_X)
                train_y = np.array(train_y)
            except:
                raise TypeError("numpy.ndarray required for train_X and ..")
        d = len(np.shape(test_X))
        if d == 1:
            sample = test_X
            result = self._classify(sample, train_X, train_y)
            results.append(result)
        else:
            for i in range(len(test_X)):
                sample = test_X[i]
                result = self._classify(sample, train_X, train_y)
                results.append(result)
        return results


if __name__ == "__main__":
    train_X = [[1, 2, 0, 1, 0],
               [0, 1, 1, 0, 1],
               [1, 0, 0, 0, 1],
               [2, 1, 1, 0, 1],
               [1, 1, 0, 1, 1]]
    train_y = [1, 1, 0, 0, 0]
    clf = KNNClassifier(k=3)
    sample = [[1, 2, 0, 1, 0], [1, 2, 0, 1, 1]]
    result = clf.classify(sample, train_X, train_y)


def imgToVector(filename):
    returnVect = np.zeros((1, 1024))
    fo = open(filename, 'r')
    fr = fo.readlines()
    m = len(fr)
    for i in range(m):
        oneline = fr[i].strip()
        n = len(oneline)
        if not n:
            continue
        for j in range(n):
            returnVect[0,n*i+j] = int(oneline[j])


def loadDataset(filedir):
    trainlist = os.listdir(filedir)
    m = len(trainlist)
    data_X = np.zeros((m,1024))
    data_y = []
    for i in range(len(trainlist)):
        filename = trainlist[i]
        label_i = int(filename.split('_')[0])
        data_y.append(label_i)
        oneSample_X = imgToVector(filedir + "/" + trainlist[i])
        data_X[i,:]  = oneSample_X
    return data_X, data_y
    

def handwritingClassTest(train_X, train_y, test_X, test_y):
    clf = KNNClassifier(k=3)
    len_test = len(test_y)
    error_count = 0.0
    results = clf.classify(test_X, train_X, train_y)
    arr_res = np.array(results) - np.array(test_y)
    for elem in arr_res:
        error_count += np.abs(elem) 
    error_rate = error_count/len_test
    print(error_rate)
    return error_rate

def main():
    traindir = "data/trainingDigits"
    testdir = "data/testDigits"
    train_X, train_y = loadDataset(traindir)
    test_X, test_y = loadDataset(testdir)
    handwritingClassTest(train_X, train_y, test_X, test_y)



if __name__=="__main__":
    main()