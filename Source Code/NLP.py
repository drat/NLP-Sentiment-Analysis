'''
[info] Installing unidecode to encode the text for nltk toolkit
[info] Install nltk.download() to be able to run this as it uses the NLTK toolkit

'''
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
# ----------------
# Classifier
from Classifier import Classifier

class Simulation(object):
	''' 
	Description: A class to train the tweet classification system and expose to live data
	'''
	
	def __init__(self, positiveVocab, negativeVocab):
		self.positiveVocab = positiveVocab 
		self.negativeVocab = negativeVocab
		self.documents = []
		self.wordCorpus = []
		self.wordFeatures = None
		self.training_set = None
		self.testing_set = None
		self.trainedEnsembleClassifier = None
	
	def extractFeatures(self,document):
		''' 
		Description: A function to extract features from text provided
		'''
		words = word_tokenize(document)
		features = {}
		for w in self.wordFeatures:
			features[w] = (w in words)
		return features
		
	def sentiment(self,tweet):
		''' 
		Description: Sentiment function that is called and fed to the trained algorithms for classification 
		'''
		thisTweetsFeatures = self.extractFeatures(tweet)
		return self.trainedEnsembleClassifier.classify(thisTweetsFeatures),float(self.trainedEnsembleClassifier.confidence(thisTweetsFeatures))
		
	def createTrainTestData(self):
		''' 
		Description: A function to create training and test data for the algorithms 
		'''
		# Get positive vocab
		for r in self.positiveVocab.split('\n'):
			self.documents.append( (r, "pos") )
		# Get negative vocab
		for r in self.negativeVocab.split('\n'):
			self.documents.append( (r, "neg") )
		# Tokinise the positive and negative words
		positiveVocab_words = word_tokenize(self.positiveVocab)
		negativeVocab_words = word_tokenize(self.negativeVocab)
		# For each positive word append the word (lower case) to the master corpus
		for w in positiveVocab_words:
			self.wordCorpus.append(w.lower())
		# For each negative word append the word (lower case) to the master corpus
		for w in negativeVocab_words:
			self.wordCorpus.append(w.lower())
		# Master word corpus is the freq distribution
		self.wordCorpus = nltk.FreqDist(self.wordCorpus)
		# We have a large training set so we choose to use 50,000 features which is very large
		self.wordFeatures = list(self.wordCorpus.keys())[:50000]
		# Generate the feature set and then shuffle
		featuresets = [(self.extractFeatures(rev), category) for (rev, category) in self.documents]
		random.shuffle(featuresets) # Shuffle the sets so its not just pos / neg / pos / neg etc
		# Get test and training data split
		length_of_features = len(featuresets)
		print '[info] Length of feature set: %s' % length_of_features
		fractions = int(length_of_features / 3)
		split_1 = fractions * 2
		split_2 = fractions * 1
		# Print to screen the split that is being implemented
		print "[info] Train on: %s" % split_1
		print "[info] Test on: %s" % split_2
		# Assign test and training sets 
		self.training_set = featuresets[split_1:]
		self.testing_set =  featuresets[:split_2]
		print "[info] Completed generating test and training datasets"
		
	def trainAlgorithms(self):
		''' 
		Description: A class to train the respective classifiers for this NLP system 
		NOTES: 
			[1] There are 7 classifiers each chosen to add a branch of intelligence to the classifier
			[2] Naive Bayey is good for short text whereas the SVM's are used as they are often very accurate
			[3] A mix of algos are added to diversify the master classifier
		'''
		# Base line algorithm - Naive Bayes
		classifier = nltk.NaiveBayesClassifier.train(self.training_set)
		print("[algorithm] Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, self.testing_set))*100)
		classifier.show_most_informative_features(15)
		# Multinomial Naive Bayes
		MNB_classifier = SklearnClassifier(MultinomialNB())
		MNB_classifier.train(self.training_set)
		print("[algorithm] MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, self.testing_set))*100)
		# Bernoulli Naive Bayes
		BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
		BernoulliNB_classifier.train(self.training_set)
		print("[algorithm] BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, self.testing_set))*100)
		# Logistic Regression
		LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
		LogisticRegression_classifier.train(self.training_set)
		print("[algorithm] LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, self.testing_set))*100)
		# Stochastic Gradient Descent CLassifier
		SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
		SGDClassifier_classifier.train(self.training_set)
		print("[algorithm] SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, self.testing_set))*100)
		# Linear Support Vector Machine (2 Support Vectors)
		LinearSVC_classifier = SklearnClassifier(LinearSVC())
		LinearSVC_classifier.train(self.training_set)
		print("[algorithm] LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, self.testing_set))*100)
		# Support Vector Machine (3 Support Vectors)
		NuSVC_classifier = SklearnClassifier(NuSVC())
		NuSVC_classifier.train(self.training_set)
		print("[algorithm] NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, self.testing_set))*100)
		# Master object which feeds the algoirhtms to the Classifier class
		self.trainedEnsembleClassifier = Classifier( classifier,NuSVC_classifier,LinearSVC_classifier,SGDClassifier_classifier,
										        MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier )
		print("[classifier] trainedEnsembleClassifier accuracy percent:", (nltk.classify.accuracy(self.trainedEnsembleClassifier, self.testing_set))*100)
	
	

def main():
	# The path to the positive and negative pre categorised training sets (root) 
	positiveVocab = open("positive.txt","r").read()
	negativeVocab = open("negative.txt","r").read()
	# Encode the file to something that can be parsed
	positiveVocab = unicode(positiveVocab, errors='ignore')
	negativeVocab = unicode(negativeVocab, errors='ignore')
	
	# Create the simulation & train algos
	print '[info] Creating simulation object'
	Sim = Simulation( positiveVocab, negativeVocab )
	# Create data for the algos
	Sim.createTrainTestData()
	# Train the algos for the classifier
	Sim.trainAlgorithms()
	# Output confomation to screen 
	print '[info] Completed Training'
	# Here we will go through all the objects and append them to the data_collection 
	tweet_path = r'C:\Users\Andrew\Documents\IRDM\GitHub\irdm_twitter_sentiment\NLP\JSON_FILES'
	# Returns a bunch of tweet objects
	tweets_objects = load_tweets(tweet_path)
	# Get number of objects so we can have some kind of % done indicator 
	total_tweets = len(tweets_objects)
	# File path where the classicied tweets are dumped along with the error file
	file_path = 'C:\Users\Andrew\Documents\IRDM\GitHub\irdm_twitter_sentiment\NLP'
	print '[info] Initiating tweet object loop through'
	# Master loop where the classification is called
	for instance in range(0,len(tweets_objects)):
		try:
			#Print progress every 2.5% of the London tweet dataset given
			if(instance % 48704 == 0):
				print (float(instance)/float(total_tweets))*100
			# Generate the classification
			classify = Sim.sentiment(tweets_objects[instance].text.encode('utf-8'))
			# Pos / Neg
			category = classify[0]
			# Confidence level
			confidence = classify[1]
			# If there is no such classificetion file on the system - create one to store the classified tweets to
			if not os.path.exists( file_path + '\\' +'classified_tweets.txt'):
				open( file_path + '\\' + 'classified_tweets.txt', 'w').close # Creates the error log file
			# Write the classified tweet to disk with its :
			# [ Classification, Confidence, ID, Tweet, Username, Datetime, UNIX, Lat, Long ]
			with open( file_path + '\\' + 'classified_tweets.txt', 'a' ) as outfile:
				outfile.write( str(category) + ',' ) # Sentiment Classification
				outfile.write( str(confidence) + ',' ) # How sure we are of this classification
				outfile.write( str(tweets_objects[instance].tweet_id) + ',' ) # Tweet ID
				outfile.write( str(tweets_objects[instance].text.encode('utf-8')) + ',' ) # Tweet
				outfile.write( str(tweets_objects[instance].username.encode('utf-8')) + ',' ) # Username
				outfile.write( str(tweets_objects[instance].timestamp) + ',' ) # Datetime 
				outfile.write( str(tweets_objects[instance].raw_unix) + ',' ) # Raw Uix Timestamp
				outfile.write( str(tweets_objects[instance].latitude) + ',' ) # Latitude
				outfile.write( str(tweets_objects[instance].longitude) + '\n' ) # Longitude
		except:
			# Any issues with classification (e.g emojies or such like are written to an error file)
			if not os.path.exists( file_path + '\\' + 'error_log.txt'):
				open( file_path + '\\' + 'error_log.txt', 'w').close # Creates the error log file
			# Output necessary details to the error log so we can see what type of tweets the system is struggling with		
			with open( file_path + '\\' + 'error_log.txt', 'a' ) as outfile:
				outfile.write( 'Tweet ID: ' + str(tweets_objects[instance].tweet_id) + '\n' )
				outfile.write( 'Tweet: ' + str(tweets_objects[instance].text.encode('utf-8')) + '\n' )
				outfile.write( 'Username: ' + str(tweets_objects[instance].username.encode('utf-8')) + '\n' )
				outfile.write( 'Timestamp (Unix): ' + str(tweets_objects[instance].raw_unix) + '\n' )
				outfile.write( 'Date: ' + str(tweets_objects[instance].timestamp) + '\n' )
				outfile.write( 'latitude: ' + str(tweets_objects[instance].latitude) + '\n' )
				outfile.write( 'longitude: ' + str(tweets_objects[instance].longitude) + '\n' )
				outfile.write( '-------------------------------------------' + '\n' )
			
if __name__ == "__main__":
	main()
