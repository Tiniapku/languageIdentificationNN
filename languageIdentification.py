# -*- coding: utf-8 -*-
import sys
from random import randint
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing.label import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as pl
np.set_printoptions(threshold=np.nan)
class languageIdentification(object):
    """
    Using characters as features, each encoded by sklearn OneHotEncoder
    Languages are encoded into vectors using sklearn LabelBinarizer
    """
    def __init__(self, trainFile, d = 100, yita = 0.1):
        self.d = d
        self.yita = yita
        self.languages = {"ENGLISH": 1, "FRENCH": 3, "ITALIAN": 2}
        self.punctuations = [".", "'", ":", ",", "-", "...", "!", "_", "(", ")", "?", '"', ";", "/", "\\", "{", "}", \
                             "[", "]", "|", "<", ">", "+", "=", "@", "#", "$", "%", "^","&", "*"]
        self.noPunctuation = False
        self.answerLables = LabelBinarizer()

        self.c = set()
        self.trainLabels = preprocessing.LabelEncoder()
        self.rawResult = []
        self.train, self.result, self.inputlen = self.trainProcessing(trainFile)
        self.v = preprocessing.OneHotEncoder(n_values=self.inputlen)
        self.train = self.v.fit_transform(self.train).toarray()
        self.input = self.inputlen * 5 + 1
        self.hidden = d
        self.output = 3

        self.ai = np.array([1.0] * self.input)
        self.ah = np.array([1.0] * (self.hidden + 1))
        self.ao = [1.0] * self.output

        self.wi = np.random.uniform(size = (self.input, self.hidden))
        self.wo= np.random.randn(self.hidden + 1, self.output)

        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden + 1, self.output))

    def trainProcessing(self, trainFile):
        trainList = []
        result = []
        #self.answerLables.fit(["ENGLISH", "FRENCH", "ITALIAN"])
        self.answerLables.fit([1,2,3])
        #print self.answerLables.classes_
        for line in trainFile:
            #print line
            space = line.index(" ")
            answer, train = line[:space], line[space + 1:]
            li, ans = self.lineProc(train, answer, True)
            trainList += li
            result += ans
        trainList, result = self.FisherYatesShuffle(trainList, result)
        self.rawResult = result
        trainList = np.array(trainList)
        trainList = self.trainLabels.fit_transform(trainList.ravel()).reshape(*trainList.shape)
        result = self.answerLables.fit_transform(result)
        #print self.trainLabels.classes_
        inputLen = len(list(self.trainLabels.classes_))
        #print trainList.shape
        #print inputLen
        #print self.v.n_values
        #print self.v.feature_indices_
        #print result
        #print trainList
        return (trainList, result, inputLen)

    def testProcessing(self, testFile):
        prediction_results = []
        for line in testFile:
            test = self.lineProc(line,"", False)
            #for i in xrange(len(test)):
            test = np.array(test)
            test = self.trainLabels.transform(test.ravel()).reshape(*test.shape)
            #print test
            test = self.v.transform(test).toarray()
            #print res
            #print res.shape
            res = self.predict(test)
            #print line, res
            prediction_results.append(res)


        return prediction_results


    def lineProc(self, line, answer, isTraining = True):
        #line = encode(line)
        text = []
        result = []
        for ch in line:
            self.c.add(ch)
        for i in xrange(len(line) - 5):
            text.append(list(line[i : i + 5]))
            if isTraining:
                result.append(self.languages[answer])
        if isTraining:
            return (text, result)
        else:
            return text

    def FisherYatesShuffle(self, train, result):
        l = len(train)
        for i in xrange(l - 1, 0, -1):
            j = randint(0, i)
            train[i], train[j] = train[j], train[i]
            result[i],result[j] = result[j], result[i]
        #print result
        return train[:], result[:]

    def feedForward(self, inputs):
        for i in range(self.input - 1):
            self.ai[i] = inputs[i]

        self.ah[:self.hidden] = np.dot(self.ai, self.wi)
        self.ah = self.sigmoid(self.ah)
        self.ah[-1] = 1

        self.ao = np.dot(self.ah, self.wo)

        self.softMax(self.ao)
        #print self.ao
        return self.ao[:]

    def softMax(self, out):
        total = sum(np.exp(out))
        for i in xrange(self.output):
            out[i]  = np.exp(out[i]) * 1.0 / total

    def backPropagate(self, result):
        # p(L, y) = y - y_hat
        d4 = self.ao - np.array(result)
        # kronecker delta: P(L, y_hat) = P(L, y) * P(y, y_hat)
        d3 = np.array([0.0] * self.output)
        for j in xrange(self.output):
            for i in xrange(self.output):
                if i == j:
                    d3[j] += d4[i] * self.ao[i] * (1 - self.ao[j])
                else:
                    d3[j] += d4[i] * self.ao[i] * -self.ao[j]
        # p(L, ah) = P(L, y) * P(y, y_hat) * p(y_hat, ah)
        #d2 = np.dot(self.wo, d3)
        d2 = np.dot(self.wo, d3)
        # p(L, ah_hat) = p(L, y) * P(y, y_hat) * p(y_hat, ah) * P(ah, ah_hat)
        d1 = d2 * self.partialDerivativeSigmoid(self.ah)
        # p(L, W2) = p(L, y) * p(y, y_hat) * p(y_hat, W2)
        D2 = self.yita * np.outer(self.ah, d3)
        self.wo -= D2 + self.co
        self.co = D2
        # p(L, w1) = p(L, y) * P(y, y_hat) * p(y_hat, ah) * P(ah, ah_hat) * P(ah_hat, w1)
        D1 = self.yita * np.outer(self.ai, d1[1:])
        self.wi -= D1 + self.ci
        self.ci = D1
        error =  np.dot(d4, d4)

        return error

    def trainNN(self, epoch = 3):
        for i in xrange(epoch):
            error = 0.0
            trainres = []
            for j in xrange(len(self.train)):
                entry = self.train[j]
                res = self.result[j]
                #print "----------------------------sentence:",res, "-------------------------"
                r = self.feedForward(entry)
                idx = np.argmax(r) + 1
                trainres.append(idx)
                error += self.backPropagate(res)
            accuracy = self.evaluate(trainres, self.rawResult)
            return accuracy, error

    def predict(self, test):
        result = Counter()
        #test = self.v.transform(test).toarray()
        #print test
        for entry in test:
            r = self.feedForward(entry)
            #print r
            idx = np.argmax(r) + 1
            result[idx] += 1

        return result.most_common(1)[0][0]

    def partialDerivativeError(self, target, out):
        #print target, out
        return out - target

    def partialDerivativeSigmoid(self, out):
        return self.sigmoid(out) * 1.0 * (1 - self.sigmoid(out))

    def sigmoid(self, x):

        return 1.0 / (1 + np.exp(x))

    def squaredLoss(self, y, y_hat):

        return abs(y - y_hat)**2 * 1.0 / 2

    def evaluate(self, predictions, golden):
        res = accuracy_score(golden, predictions)
        #print "accuracy: ", res
        #print "precision:", precision_score(golden, predictions)
        #print "Recall: ", recall_score(golden, predictions)
        return res

    def debugtestProcessing(self, testFile):
        test_results = []
        prediction_results = []
        for line in testFile:
            space = line.index(" ")
            answer, train = line[:space], line[space + 1:]
            #print answer, train
            test_results.append(self.languages[answer.strip()])
            test = self.lineProc(train,"", False)
            #for i in xrange(len(test)):
            test = np.array(test)
            test = self.trainLabels.transform(test.ravel()).reshape(*test.shape)
            #print test
            test = self.v.transform(test).toarray()
            #print res
            #print res.shape
            res = self.predict(test)
            #print line, res
            prediction_results.append(res)


        return prediction_results, test_results

    def getFileFeatures(self, content):
        features = []
        results = []
        for line in content:
            space = line.index(" ")
            answer, train = line[:space], line[space + 1:]
            #print answer, train
            results.append(self.languages[answer.strip()])
            test = self.lineProc(train,"", False)
            #for i in xrange(len(test)):
            test = np.array(test)
            test = self.trainLabels.transform(test.ravel()).reshape(*test.shape)
            test = self.v.transform(test).toarray()
            features.append(test)

        return features, results

    def predictFeatures(self, features):
        predict_result = []
        for f in features:
            res = self.predict(f)
            predict_result.append(res)
        return predict_result

def devProcess(trainFilename, devFileName, epoch = 3):
    trainAccuracy = []
    devAccuracy = []

    ftrain = open(trainFilename, 'r')
    train = ftrain.read().strip().split("\n")
    ftrain.close()

    solution = languageIdentification(train, 100, 0.1)

    fdev = open(devFileName, 'r')
    dev = fdev.read().strip().split("\n")
    fdev.close()

    train_features, train_results = solution.getFileFeatures(train)
    train_prediction = solution.predictFeatures(train_features)
    initial_train = solution.evaluate(train_prediction, train_results)
    trainAccuracy.append(initial_train)
    print "initial train accuracy: ", initial_train

    dev_features, dev_results = solution.getFileFeatures(dev)
    dev_prediction = solution.predictFeatures(dev_features)
    initial_dev = solution.evaluate(dev_prediction, dev_results)
    devAccuracy.append(initial_dev)
    print "initial dev accuracy: ", initial_dev

    for i in xrange(epoch):
        print "************************************epoch:", i + 1, "************************************"
        trainac, err = solution.trainNN(1)
        print "train accuracy:", trainac
        trainAccuracy.append(trainac)
        dev_prediction = solution.predictFeatures(dev_features)
        devac = solution.evaluate(dev_prediction, dev_results)
        print "dev accuracy:", devac
        devAccuracy.append(devac)
    x = [0,1,2,3]
    pl.plot(x, trainAccuracy, 'r--', x, devAccuracy, 'bs')
    pl.show()

if __name__ == "__main__":
    trainFileName = sys.argv[1]
    devFileName = sys.argv[2]
    testFileName = sys.argv[3]
    resultFileName = sys.argv[4]
    devProcess(trainFileName, devFileName)

    """
    ftrain = open(trainFileName, 'r')
    solution = languageIdentification(ftrain, 3, 0.1)
    #print solution.result
    #print solution.wi
    ftrain.close()
    print "************************************train************************************"
    solution.trainNN(3)
    #print solution.wi
    ftest = open(testFileName, 'r')
    print "************************************test************************************"
    predictions = solution.testProcessing(ftest)
    print predictions
    test_results = open(resultFileName, 'r')
    golden = []
    for line in test_results.readlines():
        golden.append(solution.languages[line.strip()])
    solution.evaluate(predictions, golden)
    """

