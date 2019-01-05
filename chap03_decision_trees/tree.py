"""
Created on Jan 05, 2019.
Decision Tree, from the source code for Machine Learning in Action Chap.3.

Authon: Yunlong Qi
"""
from math import log
import operator
import matplotlib.pyplot as plt

from chap03_decision_trees import treePlotter

def createDataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def calcShannonEnt(dataset):
    """
    Calculate Shannon Entropy.
    :param dataset:
    :return:
    """
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataset(dataset, axis, value):
    """
    Return the sub-dataset with dataset[axis]==value
    The no. of features of sub-dataset is less than dataset by 1.
    """
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]  # chop out axis for splitting
            reduceFeatVec.extend(featVec[axis+1:])
            retDataset.append(reduceFeatVec)
    return retDataset


def chooseBestFeatureToSplit(dataset):
    """
    Iterate through all these features to determine the best split feature.
    :param dataset:
    :return:
    """
    numFeatures = len(dataset[0]) - 1       # The last column is used for the labels.
    baseEntropy = calcShannonEnt(dataset)   # H(D)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]   # Create a list of all the examples of this feature. i.e. the i-th column
        uniqueVals = set(featList)          # Get a set of unique values of feature[i]
        newEntropy = 0.0
        for value in uniqueVals:
            subDataset = splitDataset(dataset, i, value)
            prob = len(subDataset) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy # Calculate the info gain
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount


def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList): # All examples have the same feature.
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]   # Copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataset(dataset, bestFeat, value), subLabels)
    return myTree


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def createPlot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode("A decision node", (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode("A leaf node", (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords="axes fraction",
                            xytext=centerPt, textcoords="axes fraction",
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    # myData, labels = createDataset()
    # print(labels)
    # myTree = treePlotter.retrieveTree(0)
    # print(myTree)
    # print(classify(myTree, labels, [1, 0]))
    # print(classify(myTree, labels, [1, 1]))
    # storeTree(myTree, "classifierStorage.txt")
    # print(grabTree("classifierStorage.txt"))

    # lenses
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    for lense in lenses:
        print(lense)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    treePlotter.createPlot(lensesTree)

