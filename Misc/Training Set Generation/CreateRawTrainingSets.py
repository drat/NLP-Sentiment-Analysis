'''
 Description: This is a script to parse the master tweet database which was pre-classified into pos / neg.
              It is used to split the data into the positive and negative training sets for the classification
			  algorithms.

 Parse twitter corpus - http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/
 Need this to get examples of pos / neg tweets
 1.5 million examples provided

'''
import csv
import os


class CreateTrainingData(object):
	''' A class to create our sentiment training sets '''
	
	def __init__( self ):
		self.path = 'C:\Users\Andrew\Documents\IRDM\GitHub\irdm_twitter_sentiment'

	def write_to_file( self, sentiment, tweet ):
		''' A function to write the training sets'''
		print '# Initiating write'
		if (sentiment == '0'):
			# Negative
			if not os.path.exists( self.path + '\\training_set_gen\\' + 'negative.txt'):
				print '# Creating negative.txt'
				open( self.path + '\\training_set_gen\\' +'negative.txt', 'w').close
				with open( self.path + '\\training_set_gen\\' + 'negative.txt', 'a' ) as file:
					file.write(tweet + '\n')
			else:
				with open( self.path + '\\training_set_gen\\' + 'negative.txt', 'a' ) as file:
					file.write(tweet + '\n')
		elif(sentiment == '1'):
			# Positive
			if not os.path.exists( self.path + '\\training_set_gen\\' + 'positive.txt'):
				print '# Creating positive.txt'
				open( self.path + '\\training_set_gen\\' +'positive.txt', 'w').close
				with open( self.path + '\\training_set_gen\\' + 'positive.txt', 'a' ) as file:
					file.write(tweet + '\n')
			else:
				with open( self.path + '\\training_set_gen\\' + 'positive.txt', 'a' ) as file:
					file.write(tweet + '\n')

	def deploy( self ):
		''' A function to index the data file and then split '''
		reader = csv.reader(open("Sentiment_Training_Dataset.csv"), delimiter=',')
		data = []
		for row in reader:
			data.append(row)
		for instance in range(0,len(data)):
			tweet = data[instance][3]
			sentiment = data[instance][1]
			print '-----------------------------------'
			print '## Analyse sentiment'
			if(sentiment == '0'):
				# Negative
				self.write_to_file( sentiment, tweet )
				print '# Commit negative tweet'
			elif(sentiment == '1'):
				# Positive
				self.write_to_file( sentiment, tweet )
				print '# Commit positive tweet'
			print '# Complete'
			print '-----------------------------------'
			
def main():
	obj = CreateTrainingData()
	obj.deploy()

if __name__ == "__main__":
	main()
	