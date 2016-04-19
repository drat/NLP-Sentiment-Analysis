import glob
import os
import json
from Tweet import Tweet


def main(tweet_path): #, crime_path
	tweets = load_tweets(tweet_path)
	#crimes = load_crime_data(crime_path)
	tweets_with_separated_hashtags = extract_hashtags(tweets)


def extract_hashtags(tweets):
    for tweet in tweets:
        tweet.hashtags = set([i[1:] for i in tweet.text.split() if i.startswith("#")])

def load_crime_data(path):
    crimes = []
    counter = 0

    for root, dirs, files in os.walk(path):
         for file in files:
             print root, file
             with open(os.path.join(root, file), "r") as auto:
                 print "hello"
                # load data
                # loop through each line
                # place each line into crime object.
                # crimes.append(crime)
    #return crimes

def load_tweets(path):
    tweets = []
    num_files = len(glob.glob1(path, "*.txt"));
    counter = 0
    for filename in glob.glob(os.path.join(path, '*.txt')):
        if counter % 250 == 0:
            print "Parsing file: " + str(counter) + "/" + str(num_files)

        counter += 1
        with open(os.path.join(path, filename)) as data_file:
			try:
				data = json.load(data_file)["tweets"]
				for entry in data:
					tweet = Tweet(entry["country"], entry["date"], entry["latitude"], entry["longitude"],
								  entry["placeName"], entry["placeType"], entry["profileLocation"], entry["text"],
								  entry["tweetId"], entry["username"])
					tweets.append(tweet)
			except:
				pass
    print "Parsing complete"
    return tweets


if __name__ == "__main__":
    # Place the folder containing your twitter data as the first parameter (leave the r in)
    # Place the folder containing your police data as the second parameter (leave the r in)
    main(r'C:\Users\Andrew\Downloads\Twitter')
	