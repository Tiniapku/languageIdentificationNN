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

        self.Initialize(trainFile, devFile, testFile)

        self.input = len(self.c) * 5 + 1
        self.setParameters(d, yita)
    
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

        with open(testFileName) as testFile:
            for line in testFile:
                if not line:
                    continue
                line = line.decode('latin-1').strip()
                test = self.lineProc(line, "", False)
                self.testFeatures.append(test)

        trainList, trainResult = self.FisherYatesShuffle(trainList, trainResult)
        trainResult = np.array(trainResult)
        self.trainResult = self.answerLables.fit_transform(trainResult)

        self.trainLabels = preprocessing.LabelEncoder()
        featureList = list(self.c)

        self.trainLabels.fit(featureList)
        #print self.trainLabels.classes_
        length = len(self.c)
        print "feature length:", length
        self.v = preprocessing.OneHotEncoder(n_values=length)

        trainList = np.array(trainList)
        self.train = self.trainLabels.transform(trainList.ravel()).reshape(*trainList.shape)

        self.train = self.v.fit_transform(self.train).toarray()
        print "train shape", self.train.shape

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
            initial_train = self.directPredict(self.trainFeatures, "train")[1]
            trainAccuracy.append(initial_train)
            print "initial train accuracy: ", initial_train

            initial_dev = self.directPredict(self.devFeatures, "dev")[1]
            print "initial dev accuracy: ", initial_dev
            devAccuracy.append(initial_dev)

        for i in xrange(epoch):
            print "************************************epoch:", i + 1, "************************************"
            self.trainNN(1)
            trainac = self.directPredict(self.trainFeatures, "train")[1]
            print "train accuracy:", trainac
            trainAccuracy.append(trainac)

            devac = self.directPredict(self.devFeatures, "dev")[1]
            print "dev accuracy:", devac
            devAccuracy.append(devac)

        if initial:
            x = [i for i in xrange(epoch + 1)]
            pl.plot(x, trainAccuracy, 'r--', x, devAccuracy, 'bs')
            pl.show()

    def getTestResult(self):
        test_results = open('languageIdentification.data/test_solutions', 'r')
        self.testResult = []
        for line in test_results.readlines():
            self.testResult.append(solution.languages[line.strip().split(" ")[1].upper()])

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

    def resetParameters(self):
        self.ai = np.array([1.0] * self.input)
        self.ah = np.array([1.0] * (self.hidden + 1))
        self.ao = [1.0] * self.output

        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden + 1, self.output))

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
            for j in xrange(len(self.train)):
                entry = self.train[j]
                res = self.trainResult[j]
                self.feedForward(entry)
                error += self.backPropagate(res)
            print "error:", error
            self.resetParameters()

    def predict(self, test):
        result = Counter()
        for entry in test:
            r = self.feedForward(entry)
            #print r
            idx = np.argmax(r) + 1
            result[idx] += 1

        return result.most_common(1)[0][0]

    def partialDerivativeSigmoid(self, out):
        return out * 1.0 * (1.0 - out)

    def sigmoid(self, x):
        #x =  np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    def evaluate(self, predictions, golden):
        return accuracy_score(golden, predictions)

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

if __name__ == "__main__":
    trainFileName = sys.argv[1]
    devFileName = sys.argv[2]
    testFileName = sys.argv[3]

    solution = languageIdentification(trainFileName, devFileName, testFileName, d = 100, yita = 0.1)

    print "************************************Part 0: dev data result************************************"
    solution.devProcess(30, True)
    

    print "************************************Part 1: test data result************************************"
    solution.trainNN(3)
    solution.getTestResult()
    #solution.testResultOutput(testFileName, solution.testResult)
    prediction, accuracy = solution.directPredict(solution.testFeatures, "test")
    print "test accuracy", accuracy
    if accuracy > 0.75:
        solution.testResultOutput(testFileName, prediction)
    
    print "************************************Part2: hyperparameter optimization************************************"

    d_list = [20, 50, 80, 100]
    yita_list = [0.01, 0.03, 0.1, 0.3]
    solution.getTestResult()
    for d in d_list:
        for yita in yita_list:
            solution.setParameters(d, yita)
            solution.trainNN(3)
            accuracy = solution.directPredict(solution.devFeatures, "dev")[1]
            print "The test result of hyperparameter d:", d, "and yita:", yita, accuracy

    best_d = 80
    best_yita = 0.05
    #solution.getTestResult()
    solution.setParameters(best_d, best_yita)
    for i in xrange(3):
        print "testing..."
        solution.trainNN(3)
        accuracy = solution.directPredict(solution.testFeatures, "test")[1]
        print accuracy
