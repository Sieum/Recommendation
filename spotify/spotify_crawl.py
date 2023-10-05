from crawling.settings import DATA_DIR ## 데이터셋 경로
from decouple import config ## .env 파일 읽기

import pymongo
import random

mongo_uri = f"mongodb://{config('DB_ID')}:{config('DB_PWD')}@j9a605.p.ssafy.io:27017/sieum?authSource=admin&retryWrites=true&w=majority"
client = pymongo.MongoClient(mongo_uri)
db = client["sieum"]

genre_collection = db["genre"]
music_collection = db["music"]

def recommend_by_genre(genre_list):

    recommended_music_list = []
    recommended_mudic_ids = []
    
    for genre in genre_list:
        matching_artists = genre_collection.find({ 'genres': {'$regex': genre} })
        matching_artists = list(matching_artists)
        
        if matching_artists:
            random_artist = random.choice(matching_artists)["artists"]
            matching_music = music_collection.find({"artists": {"$regex": random_artist}})
            matching_music = list(matching_music)

            if matching_music:
                random_music = random.choice(matching_music)
                recommended_music_list.append(random_music)
    
    return recommended_music_list