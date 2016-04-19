# ----------------
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


class Classifier(ClassifierI):
	''' 
	Description: A class to implement the ensemble voting sentiment analysis
	'''
    def __init__(self, *classifiers):
		self.classifiers = classifiers

    def classify(self, features):
		''' 
		Description: A function to compute and store the votes of each classifier
		'''
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
		''' 
		A function to compute the confidence in line with the results from classify
		'''
		votes = []
		for c in self.classifiers:
			v = c.classify(features)
			votes.append(v)
		mostCommonVoteCount = votes.count(mode(votes))
		generateConfidence = float(mostCommonVoteCount) / float(len(votes))
		return generateConfidence
		

