'''
Description: This script is designed to pickle and searilise the algorithms. The script
             was designed as a test to experiment if this would increase the speed in
			 which we could test and develope classifier combinations. 

NOTES:
	[1] This script did NOT get used in the final project results. The reason being that the
	    computational power required to pickle the algorithms was too great at the time of
        simulation. 
	[2] It should be noted in a commercial manner it owould be wise to pickle the algorithms
	    when the optimal parameters have been uncovered as it will allow much faster classification.
	[3] installing unidecode
	[4] install nltk.download()
'''

# Packages
import random
import pickle
import nltk
import json
import os
# ----------------
import glob
from Tweet import Tweet
from Main import extract_hashtags, load_crime_data, load_tweets
# ----------------
# String Encoding
from unidecode import unidecode
# ----------------
# NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI #so we can inherit from the nltk classifier class
from nltk.tokenize import word_tokenize
# ----------------
# Machine Learning
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
# ----------------
# Statistics
from statistics import mode #how we are going to choose who got the most votes

# Storage containers
documents = []
wordCorpus = []

def extractFeatures(document):
	''' A function to extract a documents (tweets) features for analysis '''
    words = word_tokenize(document)
    features = {}
    for w in wordFeatures:
        features[w] = (w in words)
    return features

print '[info] Loading data'	   
positiveData = open("Training/positive.txt","r").read()
negativeData = open("Training/negative.txt","r").read()

print '[info] Encoding data...'
# Encode the file to something that can be parsed - 
# sometimes issues occur with emojies for example or
# arabic language seems to cause issues - any errorous
# classifications that did not complete are written to
# and error log
positiveData = unicode(positiveData, errors='ignore')
negativeData = unicode(negativeData, errors='ignore')
# Print to screen training has been initiated
print '[info] Training in progress'

# J is adject, R is adverb, and V is verb
# We allow all of them - although further development should look to
# experiment if for example only using verbs as a feature is more effective
allowed_word_types = ["J","R","V"]
# Get possitive words for the corpus
for p in positiveData.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            wordCorpus.append(w[0].lower())
# Get negative words for the corpus
for p in negativeData.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            wordCorpus.append(w[0].lower())
# Save pickled documents
saveDocuments = open("Pickled/Documents.pickle","wb")
pickle.dump(documents, saveDocuments)
saveDocuments.close()
# Create word corpus of all words
wordCorpus = nltk.FreqDist(wordCorpus)
# Select the 50,000 most common words as the features
wordFeatures = list(wordCorpus.keys())[:50000] 
# Save word features - these are essentially the identity of positive or negative words 
saveWordFeatures = open("Pickled/wordFeatures50k.pickle","wb")
pickle.dump(wordFeatures, saveWordFeatures)
saveWordFeatures.close()
# Create feature set
featuresets = [(extractFeatures(rev), category) for (rev, category) in documents]
# Pickle the feature set
save_featuresets = open("Pickled/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
saveWordFeatures.close()
# Shuffle feature set to avoid repetitive neg,pos,neg,pos etc ...
random.shuffle(featuresets)
# 2/3's of data for training
trainingDataset = featuresets[:30000]
# 1/3 for testing
testingDataset =  featuresets[30000:]

# --------------------------- SAVE TRAINED ALGOS ---------------------------
# --------------------------- CAN USE THEM LATER ---------------------------

classifier = nltk.NaiveBayesClassifier.train(trainingDataset)
print("[algorithm] Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testingDataset))*100)
classifier.show_most_informative_features(25)

##
saveThisClassifier = open("Pickled/NaiveBayes50k.pickle","wb")
pickle.dump(classifier, saveThisClassifier)
saveThisClassifier.close()
##

# ----------------------------------------------------

MultinomialNB_Classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_Classifier.train(trainingDataset)
print("[algorithm] MultinomialNB_Classifier accuracy percent:", (nltk.classify.accuracy(MultinomialNB_Classifier, testingDataset))*100)

##
saveThisClassifier = open("Pickled/MultinomialNB_Classifier50k.pickle","wb")
pickle.dump(MultinomialNB_Classifier, saveThisClassifier)
saveThisClassifier.close()
##

# ----------------------------------------------------

BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(trainingDataset)
print("[algorithm] BernoulliNB_Classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_Classifier, testingDataset))*100)

##
saveThisClassifier = open("Pickled/BernoulliNB_Classifier50k.pickle","wb")
pickle.dump(BernoulliNB_Classifier, saveThisClassifier)
saveThisClassifier.close()
##

# ----------------------------------------------------

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(trainingDataset)
print("[algorithm] LogisticRegression_Classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_Classifier, testingDataset))*100)

##
saveThisClassifier = open("Pickled/LogisticRegression_Classifier50k.pickle","wb")
pickle.dump(LogisticRegression_Classifier, saveThisClassifier)
saveThisClassifier.close()
##

# ----------------------------------------------------

StochasticGradientDescent_Classifier = SklearnClassifier(SGDClassifier())
StochasticGradientDescent_Classifier.train(trainingDataset)
print("[algorithm] StochasticGradientDescent_Classifier accuracy percent:", (nltk.classify.accuracy(StochasticGradientDescent_Classifier, testingDataset))*100)

##
saveThisClassifier = open("Pickled/SGDC_classifier50k.pickle","wb")
pickle.dump(StochasticGradientDescent_Classifier, saveThisClassifier)
saveThisClassifier.close()
##

# ----------------------------------------------------

LinearSupportVector_Classifier = SklearnClassifier(LinearSVC())
LinearSupportVector_Classifier.train(trainingDataset)
print("[algorithm] LinearSupportVector_Classifier accuracy percent:", (nltk.classify.accuracy(LinearSupportVector_Classifier, testingDataset))*100)

##
saveThisClassifier = open("Pickled/LinearSupportVector_Classifier50k.pickle","wb")
pickle.dump(LinearSupportVector_Classifier , saveThisClassifier)
saveThisClassifier.close()
##

# ----------------------------------------------------

NuSupportVector_Classifier = SklearnClassifier(NuSVC())
NuSupportVector_Classifier.train(trainingDataset)
print("[algorithm] NuSupportVector_Classifier accuracy percent:", (nltk.classify.accuracy(NuSupportVector_Classifier, testingDataset))*100)

##
saveThisClassifier = open("Pickled/NuSupportVector_Classifier50k.pickle","wb")
pickle.dump(NuSupportVector_Classifier, saveThisClassifier)
saveThisClassifier.close()
##

# ----------------------------------------------------

print '[info] Training completed'
print '[info] Pickling completed'