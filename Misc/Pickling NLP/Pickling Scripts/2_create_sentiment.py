
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
# ----------------
# Voting Classifier Class 
from EnsembleClassifier import EnsembleClassifier


def extractFeatures(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Get pickled documents
pickledDocs = open("Pickled/Documents.pickle", "rb")
documents = pickle.load(pickledDocs)
pickledDocs.close()
# Get pickled word features
wordPickle = open("Pickled/wordFeatures50k.pickle", "rb")
wordFeatures = pickle.load(wordPickle)
wordPickle.close()
# Get pickled feature sets
pickledFeaturesets = open("Pickled/featuresets.pickle", "rb")
featuresets = pickle.load(pickledFeaturesets)
pickledFeaturesets.close()
# Shuffle the feature set
random.shuffle(featuresets)

## LOAD THE ALGOS FROM PICKLED STATE

open_file = open("Pickled/NaiveBayes50k.pickle", "rb")
NB_Classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/MultinomialNB_Classifier50k.pickle", "rb")
MultinomialNB_Classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/BernoulliNB_Classifier50k.pickle", "rb")
BernoulliNB_Classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/LogisticRegression_Classifier50k.pickle", "rb")
LogisticRegression_Classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/SGDC_classifier50k.pickle", "rb")
StochasticGradientDescent_Classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/LinearSupportVector_Classifier50k.pickle", "rb")
LinearSupportVector_Classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/NuSupportVector_Classifier50k.pickle", "rb")
NuSupportVector_Classifier = pickle.load(open_file)
open_file.close()

## END LOADING

trainedClassifier = EnsembleClassifier( NB_Classifier,
                                        MultinomialNB_Classifier,
                                        BernoulliNB_Classifier,
                                        LogisticRegression_Classifier 
                                        StochasticGradientDescent_Classifier,
                                        LinearSupportVector_Classifier,
                                        NuSupportVector_Classifier
                                         )

def sentiment(text):
    feats = extractFeatures(text)
    return trainedClassifier.classify(feats),float(trainedClassifier.confidence(feats))

# Here we will go through all the objects and append them to the data_collection 
tweet_path = r'C:\Users\Andrew\Downloads\Twitter'
# Returns a bunch of objects
tweets_objects = load_tweets(tweet_path)
# File Dumping path
file_path = 'C:\Users\Andrew\Documents\IRDM\GitHub\irdm_twitter_sentiment\NLP\Optimised'
for instance in range(0,len(tweets_objects)):
	try:
		classify = sentiment(tweets_objects[instance].text.encode('utf-8'))
		category = classify[0]
		confidence = classify[1]
		if not os.path.exists( file_path + '\\' +'classified_tweets.txt'):
			open( file_path + '\\' + 'classified_tweets.txt', 'w').close # Creates the log file
		with open( file_path + '\\' + 'classified_tweets.txt', 'a' ) as outfile:
			outfile.write( str(category) + ',' ) #Sentiment Classification
			outfile.write( str(confidence) + ',' ) #How sure we are of this classification
			outfile.write( str(tweets_objects[instance].tweet_id) + ',' ) #Tweet ID
			outfile.write( str(tweets_objects[instance].text.encode('utf-8')) + ',' ) #Tweet
			outfile.write( str(tweets_objects[instance].username.encode('utf-8')) + ',' ) #Username
			outfile.write( str(tweets_objects[instance].timestamp) + ',' ) #Datetime 
			outfile.write( str(tweets_objects[instance].raw_unix) + ',' ) #raw_unix
			outfile.write( str(tweets_objects[instance].latitude) + ',' ) #latitude
			outfile.write( str(tweets_objects[instance].longitude) + '\n' ) #longitude
		print 'classified tweet number ' + str(instance)
	except:
		## THE ERRORS CAN BE WRITTEN TO A FILE
		print 'ERROR ENCOUNTERED'
		if not os.path.exists( file_path + '\\' + 'error_log.txt'):
			open( file_path + '\\' + 'error_log.txt', 'w').close # Creates the error log file
		with open( file_path + '\\' + 'error_log.txt', 'a' ) as outfile:
			outfile.write( 'Tweet ID: ' + str(tweets_objects[instance].tweet_id) + '\n' )
			outfile.write( 'Tweet: ' + str(tweets_objects[instance].text.encode('utf-8')) + '\n' )
			outfile.write( 'Username: ' + str(tweets_objects[instance].username.encode('utf-8')) + '\n' )
			outfile.write( 'Timestamp (Unix): ' + str(tweets_objects[instance].raw_unix) + '\n' )
			outfile.write( 'Date: ' + str(tweets_objects[instance].timestamp) + '\n' )
			outfile.write( 'latitude: ' + str(tweets_objects[instance].latitude) + '\n' )
			outfile.write( 'longitude: ' + str(tweets_objects[instance].longitude) + '\n' )
			outfile.write( '-------------------------------------------' + '\n' )

