import datetime

class Tweet:
    def __init__(self, country, date, latitude, longitude, placeName, placeType, profileLocation, text, tweetId,
                 username):
        self.country = country
        self.date = datetime.datetime.fromtimestamp(int(date / 1000)).strftime('%d-%m-%Y')
        self.latitude = latitude
        self.longitude = longitude
        self.place_name = placeName
        self.place_type = placeType
        self.profile_location = profileLocation
        self.text = text
        self.tweet_id = tweetId
        self.username = username
        self.hashtags = []
        self.timestamp = datetime.datetime.fromtimestamp(int(date / 1000))
        self.raw_unix = date

