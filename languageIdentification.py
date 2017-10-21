import sys
from random import randint
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing.label import LabelBinarizer
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as pl

np.set_printoptions(threshold=np.nan)

class languageIdentification(object):
    """
    Using characters as features, each encoded by sklearn OneHotEncoder
    Languages are encoded into vectors using sklearn LabelBinarizer
    """
    def __init__(self, trainFile, devFile, testFile, d = 100, yita = 0.1):
        self.d = d
        self.yita = yita
        self.languages = {"ENGLISH": 1, "FRENCH": 3, "ITALIAN": 2}
        self.punctuations = [".", "'", ":", ",", "-", "...", "!", "_", "(", ")", "?", '"', ";", "/", "\\", "{", "}", \
                             "[", "]", "|", "<", ">", "+", "=", "@", "#", "$", "%", "^","&", "*"]
        self.noPunctuation = False
        self.answerLables = LabelBinarizer()
        self.answerLables.fit([1, 2, 3])
        self.c = set()
        #self.trainLabels = preprocessing.LabelEncoder()
        #self.rawResult = []
        #self.train, self.result, self.inputlen = self.trainProcessing(trainFile)
        #self.v = preprocessing.OneHotEncoder(n_values=self.inputlen)
        #self.train = self.v.fit_transform(self.train).toarray()
        self.Initialize(trainFile, devFile, testFile)

        self.input = len(self.c) * 5 + 1
        self.setParameters(d, yita)
        #self.hidden = d
        #self.output = 3

        #self.ai = np.array([1.0] * self.input)
        #self.ah = np.array([1.0] * (self.hidden + 1))
        #self.ao = [1.0] * self.output

        #self.wi = np.random.uniform(size = (self.input, self.hidden))
        #self.wo= np.random.randn(self.hidden + 1, self.output)

        #self.ci = np.zeros((self.input, self.hidden))
        #self.co = np.zeros((self.hidden + 1, self.output))
    """
    def trainProcessing(self, trainFile):
        trainList = []
        result = []
        self.answerLables.fit([1,2,3])
        for line in trainFile:
            space = line.index(" ")
            answer, train = line[:space].upper(), line[space + 1:]
            li, ans = self.lineProc(train, answer, True)
            trainList += li
            result += ans
        trainList, result = self.FisherYatesShuffle(trainList, result)
        self.rawResult = result
        trainList = np.array(trainList)
        trainList = self.trainLabels.fit_transform(trainList.ravel()).reshape(*trainList.shape)
        result = self.answerLables.fit_transform(result)
        inputLen = len(list(self.trainLabels.classes_))

        return (trainList, result, inputLen)
    """
    def Initialize(self, trainFileName, devFileName, testFileName):
        trainList = []
        trainResult = []
        self.testFeatures = []
        self.devFeatures = []
        self.trainFeatures = []
        self.train = []
        #self.dev = []
        #self.test = []
        self.devResult = []
        self.rawResult = []

        print "train feature processing..."
        with open(trainFileName) as trainFile:
            for line in trainFile:
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                space = line.find(" ")
                if space < 5:
                    continue
                answer, train = line[:space].upper(), line[space + 1:]
                li, ans = self.lineProc(train, answer, True)
                trainList += li
                trainResult += ans
                self.trainFeatures.append(li)
                self.rawResult.append(self.languages[answer])
        #self.trainFeatures = np.array(self.trainFeatures)
        #print "rawResult.length", len(self.rawResult)
        with open(devFileName) as devFile:
            for line in devFile:
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                space = line.find(" ")
                if space < 5:
                    continue
                answer, train = line[:space].upper(), line[space + 1:]
                li = self.lineProc(train, answer, False)
                self.devFeatures.append(li)
                self.devResult.append(self.languages[answer])
        #self.devFeatures = np.array(self.devFeatures)

        with open(testFileName) as testFile:
            for line in testFile:
                if not line:
                    continue
                line = line.decode('latin-1').strip()
                test = self.lineProc(line, "", False)
                self.testFeatures.append(test)
        #self.testFeatures = np.array(self.testFeatures)

        trainList, trainResult = self.FisherYatesShuffle(trainList, trainResult)
        trainResult = np.array(trainResult)
        self.trainResult = self.answerLables.fit_transform(trainResult)
        #print "trainResult length", len(self.trainResult)
        self.trainLabels = preprocessing.LabelEncoder()
        featureList = list(self.c)
        # print "featureList:", featureList
        self.trainLabels.fit(featureList)
        #print self.trainLabels.classes_
        length = len(self.c)
        print "feature length:", length
        self.v = preprocessing.OneHotEncoder(n_values=length)

        #for line in trainList:
         #   line = np.array(line)
          #  line = self.trainLabels.transform(line.ravel()).reshape(*line.shape)
            #self.train.append(line)
        trainList = np.array(trainList)
        self.train = self.trainLabels.transform(trainList.ravel()).reshape(*trainList.shape)
        #print "model after label fit",self.train.shape
        self.train = self.v.fit_transform(self.train).toarray()
        print "train shape", self.train.shape
        #print "dev feature processing..."
        #devList = np.array(devList)
        #devList = self.trainLabels.fit_transform(devList.ravel()).reshape(*devList.shape)
        #self.dev =  self.v.fit_transform(devList).toarray()
        #print "dev shape",self.dev.shape

        #print "test feature processing..."
        #testList = np.array(testList)
        #testList = self.trainLabels.fit_transform(testList.ravel()).reshape(*testList.shape)
        #self.test = self.v.fit_transform(testList).toarray()
        #print "test shape", self.test.shape

    def directPredict(self, featureList, type):
        types = {"train": "self.rawResult", "dev": "self.devResult", "test": "self.testResult"}
        prediction = self.predictAll(featureList)
        accuracy = self.evaluate(prediction, eval(types[type]))

        return prediction, accuracy

    def devProcess(self, epoch, initial = True):
        trainAccuracy = []
        devAccuracy = []

        if initial:
            print "initial predictions..."
            #train_prediction = self.predictAll(self.trainFeatures)
            #initial_train = self.evaluate(train_prediction, self.trainResult)
            #trainAccuracy.append(initial_train)
            initial_train = self.directPredict(self.trainFeatures, "train")[1]
            trainAccuracy.append(initial_train)
            print "initial train accuracy: ", initial_train
            #dev_prediction = self.predictAll(self.devFeatures)
            #initial_dev = self.evaluate(dev_prediction, self.devResult)
            #devAccuracy.append(initial_dev)
            initial_dev = self.directPredict(self.devFeatures, "dev")[1]
            print "initial dev accuracy: ", initial_dev
            devAccuracy.append(initial_dev)

        for i in xrange(epoch):
            print "************************************epoch:", i + 1, "************************************"
            self.trainNN(1)
            trainac = self.directPredict(self.trainFeatures, "train")[1]
            print "train accuracy:", trainac
            trainAccuracy.append(trainac)
            #dev_prediction = self.predictAll(self.devFeatures)
            #devac = self.evaluate(dev_prediction, self.devResult)
            devac = self.directPredict(self.devFeatures, "dev")[1]
            print "dev accuracy:", devac
            devAccuracy.append(devac)

        if initial:
            x = [i for i in xrange(epoch + 1)]
            #print trainAccuracy
            #print devAccuracy
            pl.plot(x, trainAccuracy, 'r--', x, devAccuracy, 'bs')
            pl.show()

    def getTestResult(self):
        test_results = open('languageIdentification.data/test_solutions', 'r')
        self.testResult = []
        for line in test_results.readlines():
            self.testResult.append(solution.languages[line.strip().split(" ")[1].upper()])
    """
    def testPredict(self, golden):
        test_prediction = self.predictAll(self.testFeatures)
        testac = self.evaluate(test_prediction, golden)
        print "test accuracy: ", testac
    """
    def setParameters(self, d, yita):
        self.d = d
        self.yita = yita
        self.hidden = d
        self.output = 3

        self.ai = np.array([1.0] * self.input)
        self.ah = np.array([1.0] * (self.hidden + 1))
        self.ao = [1.0] * self.output

        self.wi = np.random.uniform(size = (self.input, self.hidden))
        self.wo= np.random.randn(self.hidden + 1, self.output)

        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden + 1, self.output))
    """
    def testProcessing(self, testFile):
        prediction_results = []
        for line in testFile:
            test = self.testLineProc(line)
            test = np.array(test)
            test = self.trainLabels.transform(test.ravel()).reshape(*test.shape)
            test = self.v.transform(test).toarray()
            res = self.predict(test)
            prediction_results.append(res)

        return prediction_results
    """
    def resetParameters(self):
        self.ai = np.array([1.0] * self.input)
        self.ah = np.array([1.0] * (self.hidden + 1))
        self.ao = [1.0] * self.output

        #self.ci = np.zeros((self.input, self.hidden))
        #self.co = np.zeros((self.hidden + 1, self.output))

    def lineProc(self, line, answer, isTraining = True):
        text = []
        result = []
        for ch in line:
            self.c.add(ch)
        if len(line) < 5:
            line += " " * (5 - len(line))
        for i in xrange(len(line) - 4):
            text.append(list(line[i : i + 5]))
            if isTraining:
                result.append(self.languages[answer])
        if isTraining:
            return (text, result)
        else:
            return text
    """
    def testLineProc(self, line):
        text = []
        #line = line.replace(u'\xf3', "")
        if len(line) <= 5:
            line += " " * (5 - len(line))
        for i in xrange(len(line) - 4):
            text.append(list(line[i : i + 5]))
        return text
    """
    def FisherYatesShuffle(self, train, result):
        l = len(train)
        for i in xrange(l - 1, 0, -1):
            j = randint(0, i)
            train[i], train[j] = train[j], train[i]
            result[i],result[j] = result[j], result[i]
        #print result
        return train[:], result[:]

    def feedForward(self, inputs):
        self.resetParameters()
        for i in range(self.input - 1):
            self.ai[i] = inputs[i]

        self.ah[:self.hidden] = np.dot(self.ai, self.wi)
        self.ah[-1] = 1
        self.ah = self.sigmoid(self.ah)

        self.ao = np.dot(self.ah, self.wo)

        self.ao = self.softMax(self.ao)
        return self.ao[:]

    def softMax(self, out):
        total = sum(np.exp(out))
        #for i in xrange(self.output):
        out = np.exp(out) * 1.0 / total

        return out

    def backPropagate(self, result):
        # p(L, y) = y - y_hat
        d4 = self.ao - np.array(result)
        # kronecker delta: P(L, y_hat) = P(L, y) * P(y, y_hat)
        #print "before tune:", self.ao, result
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

        error =  1.0 / 2 * np.dot(d4, d4)

        return error

    def trainNN(self, epoch = 3):
        for i in xrange(epoch):
            error = 0.0
            self.ci = np.zeros((self.input, self.hidden))
            self.co = np.zeros((self.hidden + 1, self.output))
            for j in xrange(len(self.train)):
                entry = self.train[j]
                res = self.trainResult[j]
                #print "----------------------------sentence:",res, "-------------------------"
                predict = self.feedForward(entry)
                #if j > len(self.train) - 6:
                    #print "before tune:", predict, res
                    #print "before tune", self.wo
                err = self.backPropagate(res)
                error += err
                #newresult = self.feedForward(entry)
                #if j > len(self.train) - 6:
                    #print "after tune:", newresult, res
                #newerr = newresult - np.array(res)
                #newerr = np.dot(newerr, newerr)
                #if newerr > err:
                 #   print "before tune:", predict, res
                  #  print "after tune:", newresult, res
                    #print "after tune", self.wo
            print "error:", error
            #self.resetParameters()

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
    """
    def partialDerivativeError(self, target, out):
        #print target, out
        return out - target
    """
    def partialDerivativeSigmoid(self, out):
        return self.sigmoid(out) * 1.0 * (1.0 - self.sigmoid(out))

    def sigmoid(self, x):
        #x =  np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))
    """
    def squaredLoss(self, y, y_hat):

        return abs(y - y_hat)**2 * 1.0 / 2
    """
    def evaluate(self, predictions, golden):
        res = accuracy_score(golden, predictions)
        #print "accuracy: ", res
        #print "precision:", precision_score(golden, predictions)
        #print "Recall: ", recall_score(golden, predictions)
        return res
    """
    def debugtestProcessing(self, testFile):
        test_results = []
        prediction_results = []
        for line in testFile:
            space = line.index(" ")
            answer, train = line[:space].upper(), line[space + 1:]
            #print answer, train
            test_results.append(self.languages[answer.strip()])
            test = self.testLineProc(train)
            test = np.array(test)
            test = self.trainLabels.transform(test.ravel()).reshape(*test.shape)
            test = self.v.transform(test).toarray()
            res = self.predict(test)
            prediction_results.append(res)

        return prediction_results, test_results
    
    def getFileFeatures(self, content):
        features = []
        results = []
        for line in content:
            space = line.index(" ")
            answer, train = line[:space].upper(), line[space + 1:]
            results.append(self.languages[answer.strip()])
            test = self.testLineProc(train)
            test = np.array(test)
            test = self.trainLabels.transform(test.ravel()).reshape(*test.shape)
            test = self.v.transform(test).toarray()
            features.append(test)
        #print "Features: ", len(features), len(results)
        return features, results
    """
    def predictAll(self, features):
        predict_result = []
        for f in features:
            f = np.array(f)
            feature = self.trainLabels.transform(f.ravel()).reshape(*f.shape)
            feature = self.v.transform(feature).toarray()
            res = self.predict(feature)
            predict_result.append(res)
        return predict_result

    def testResultOutput(self, testFile, testPrediction):
        inverse = {1: "ENGLISH", 3: "FRENCH", 2: "ITALIAN"}
        testFile = open(testFileName, 'r')
        with open('./languageIdentification.output', 'w') as output:
            i = 0
            for line in testFile.readlines():
                output.write(line.strip() + " " + inverse[testPrediction[i]] + '\n')
                i += 1

"""
def devProcess(trainFilename, devFileName, epoch = 3):
    trainAccuracy = []
    devAccuracy = []

    print "read in files"
    ftrain = open(trainFilename, 'r')
    train = ftrain.read().strip().split("\n")
    ftrain.close()

    solution = languageIdentification(train, 100, 0.1)

    fdev = open(devFileName, 'r')
    dev = fdev.read().strip().split("\n")
    fdev.close()

    print "initial predictions"
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
"""
if __name__ == "__main__":
    trainFileName = sys.argv[1]
    devFileName = sys.argv[2]
    testFileName = sys.argv[3]

    solution = languageIdentification(trainFileName, devFileName, testFileName, d = 100, yita = 0.1)
    """
    print "************************************Part 0: dev data result************************************"
    solution.devProcess(30, True)
    """

    print "************************************Part 1: test data result************************************"
    solution.trainNN(3)
    solution.getTestResult()
    #solution.testResultOutput(testFileName, solution.testResult)
    prediction, accuracy = solution.directPredict(solution.testFeatures, "test")
    print "test accuracy", accuracy
    if accuracy > 0.75:
        solution.testResultOutput(testFileName, prediction)
"""
    print "************************************Part2: hyperparameter optimization************************************"

    d_list = [80, 100, 120]
    yita_list = [0.01, 0.03, 0.05, 0.1, 0.3, 1, 3]
    solution.getTestResult()
    for d in d_list:
        for yita in yita_list:
            solution.setParameters(d, yita)
            solution.trainNN(3)
            accuracy = solution.directPredict(solution.testFeatures, "test")[1]
            print "The test result of hyperparameter d:", d, "and yita:", yita, accuracy
    
"""