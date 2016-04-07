import numpy as np
import math
import random
import sys
import scipy.io as sio
NUMTREES = 50
MAXDEPTH = 20
ITERS = 6
#toplevel = []

def load(path):
	data = sio.loadmat(path) #training_data and training_labels
	trainFeatures = data["training_data"]
	trainLabels = np.ravel(data["training_labels"])
	testFeatures = data["test_data"]
	return trainFeatures,trainLabels,testFeatures

def splitValidationTrain(trainFeatures,trainLabels):
	train = zip(trainFeatures,trainLabels)
	random.shuffle(train)
	val = train[:(len(train)*2)/5]
	valFeatures = [i for i,j in val]
	valLabels = [j for i,j in val]
	train = train[(len(train)*2)/5:]
	trainFeatures = [i for i,j in train]
	trainLabels = [j for i,j in train]
	return valFeatures,valLabels,trainFeatures,trainLabels

def majority(lst):
	if not lst:
		return 
	return max([0,1], key = lst.count)

def split(i, thresh, trainFeatures, trainLabels):
	leftFeatures, leftLabels, rightFeatures, rightLabels = [],[],[],[]
	for x in xrange(len(trainFeatures)): #don't use whole featuresSet
		if trainFeatures[x][i] <= thresh:
			leftFeatures.append(trainFeatures[x])
			leftLabels.append(trainLabels[x])
		else:
			rightFeatures.append(trainFeatures[x])
			rightLabels.append(trainLabels[x])
	return leftFeatures,leftLabels,rightFeatures,rightLabels

def entropy(count0, count1):
	count0frac = float(count0)/(count0 + count1)
	count1frac = float(count1)/(count0 + count1)
	log0 = 0 if count0frac == 0 else count0frac*math.log(count0frac, 2)
	log1 = 0 if count1frac == 0 else count1frac*math.log(count1frac, 2)
	return  -log0 - log1

def bagging(trainFeatures, trainLabels, k):
	k = int(math.ceil(k*len(trainLabels))) #k can be anywhere between 0.7n and n
	baggedFeatures, baggedLabels = [None] * k, [None] * k	
	for i in xrange(k):
		j = int(random.random() * len(trainFeatures))
		baggedFeatures[i] = trainFeatures[j] #change parameter for build
		baggedLabels[i] = trainLabels[j]
	return baggedFeatures,baggedLabels

def buildDecisionTree((trainFeatures, trainLabels), depth):
	if trainLabels.count(trainLabels[0]) == len(trainLabels): #base case
		return (trainLabels[0], None, None, None) #node format: (class in {0,1} if leaf else index of feature, None if leaf else threshhold, None if leaf else leftChild, None if leaf else rightChild)
	
	elif depth == MAXDEPTH:
		return (majority(trainLabels), None, None, None)
	r = random.sample(xrange(len(trainFeatures[0])), int(math.ceil(math.sqrt(len(trainFeatures[0])))) + 5)
	
	# r = range(32)
	
	infoGains = []
	for i in r:
		sortedFeatI = sorted([(features[i],label) for features,label in zip(trainFeatures,trainLabels)])[::-1]
		#reverse sorted list of feature i for all training points grouped with label
		#print "sortedFeatI", sortedFeatI #delete duplicates
		lcount1 = sum([p[1] for p in sortedFeatI])
		lcount0 = len(trainLabels) - lcount1
		rcount1 = 0
		rcount0 = 0
		#maxFeatI = max([features[i] for features in trainFeatures])
		for index,(thresh,label) in enumerate(sortedFeatI):
			if index > 0 and thresh != sortedFeatI[index-1][0]:
				# assert lcount1 + lcount0 != 0
				# assert rcount1 + rcount0 != 0 
				# assert thresh != maxFeatI
				lcoeff = float(lcount0 + lcount1)/len(sortedFeatI)
				rcoeff = float(rcount0 + rcount1)/len(sortedFeatI)
				infoGains.append(((lcoeff*entropy(lcount0, lcount1)+rcoeff*entropy(rcount0, rcount1)), i, thresh))
			rcount1 += label
			rcount0 += (1 - label)
			lcount1 -= label
			lcount0 -= (1 - label)
	if len(infoGains) == 0:
		return (majority(trainLabels), None, None, None)
	infoGain,i,thresh = min(infoGains)
	if depth == 0:
		#toplevel.append((i, thresh))
	leftFeatures,leftLabels,rightFeatures,rightLabels = split(i,thresh,trainFeatures,trainLabels) #speed this up
	if not leftLabels or not rightLabels:
		return (majority(trainLabels), None, None, None)
	leftChild = buildDecisionTree((leftFeatures,leftLabels), depth + 1)
	rightChild = buildDecisionTree((rightFeatures,rightLabels), depth + 1)
	return (i, thresh, leftChild, rightChild)

def buildRandomForest(size, trainFeatures, trainLabels):
	return [buildDecisionTree(bagging(trainFeatures, trainLabels, 0.8), 0) for i in xrange(size)]

def classify(features, tree):
	if tree[1] == None:
		return tree[0]
	elif features[tree[0]] <= tree[1]:
		#print "feature", tree[0], "$\leq$ thresh", tree[1], "\\\\"
		return classify(features, tree[2])
	return classify(features, tree[3])

def test():
	trainFeatures,trainLabels,testFeatures = load("spam-dataset/spam_data.mat")
	forest = buildRandomForest(NUMTREES, trainFeatures, trainLabels)
	testLabels = open("test_labels.csv", "w")
	testLabels.write("Id,Category\n")
	for i, features in enumerate(testFeatures):
		votes = []
		for tree in forest:
			votes.append(classify(features, tree))
		prediction = majority(votes)
		testLabels.write(str(i+1) + "," + str(prediction) + "\n")
	testLabels.close()

def validate():
	trainFeatures,trainLabels,testFeatures = load("spam-dataset/spam_data.mat")
	valFeatures,valLabels,trainFeatures,trainLabels = splitValidationTrain(trainFeatures,trainLabels)
	error = 0
	forest = buildRandomForest(NUMTREES, trainFeatures, trainLabels)
	predValLabels = []
	for features in valFeatures:
		votes = []
		for tree in forest:
			#print "a"
			votes.append(classify(features, tree))
		prediction = majority(votes)
		predValLabels.append(prediction)
	error = sum([1 for i in range(len(valLabels)) if valLabels[i] != predValLabels[i]])/float(len(valLabels))
	print "error", error
	return error

#crossvalidation
# FEATURENUMS = [0, 2, 4, 6, 8]
# values = []
# for FEATURE in FEATURENUMS:
# 	FEATURES = FEATURE
# 	meanvalue = 0
# 	for i in range(ITERS):
# 		meanvalue += validate()
# 	values.append((float(meanvalue)/ITERS, FEATURES))
# print sorted(values)

validate()
# test()