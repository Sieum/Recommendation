from crawling.settings import DATA_DIR ## 데이터셋 경로
from decouple import config ## .env 파일 읽기

####

import spotipy ## 스포티파이
from spotipy.oauth2 import SpotifyClientCredentials
import pprint ## 판다스 보기 좋게 프린트
import pandas as pd ## 판다스
pd.set_option('display.max_columns', None)  ## 판다스 모든 열 출력
from sklearn.preprocessing import MinMaxScaler ## 사이킷런 데이터 정규화
import os
import warnings
warnings.filterwarnings("ignore")

import time

import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

from collections import defaultdict
import difflib

from spotify.models import Music

####

import pymongo
import random
from django.conf import settings

####

## 데이터
# data_path = os.path.join(DATA_DIR, 'data.csv')
# genre_path = os.path.join(DATA_DIR, 'data_by_genres.csv')
# year_path = os.path.join(DATA_DIR, 'data_by_year.csv')
# data_genre_path = os.path.join(DATA_DIR, 'data_w_genres.csv')

data_path = os.path.join(settings.STATICFILES_DIRS[0], 'data.csv')
genre_path = os.path.join(settings.STATICFILES_DIRS[0], 'data_by_genres.csv')
year_path = os.path.join(settings.STATICFILES_DIRS[0], 'data_by_year.csv')
data_genre_path = os.path.join(settings.STATICFILES_DIRS[0], 'data_w_genres.csv')

data = pd.read_csv(data_path)
genre_data = pd.read_csv(genre_path)
year_data = pd.read_csv(year_path)
data_w_genre_data = pd.read_csv(data_genre_path)

## 스포티파이 개발자 로그인
cid = config('SPOTIFY_ID')
secret = config('SPOTYFY_SECRET')
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

sp = spotipy.Spotify(auth_manager=client_credentials_manager)

## KMeans 군집화
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                      ('kmeans', KMeans(n_clusters=20,
                                                        verbose=False))
                                      ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)

####


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

####

def recommend_music(song_list):

    def find_song(name):
        song_data = defaultdict()
        results = sp.search(q='track: {}'.format(name), limit=1)
        if results['tracks']['items'] == []:
            return None

        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = sp.audio_features(track_id)[0]
        song_data['artists'] = [results['artists'][0]['name']]
        song_data['release_date'] = [results['album']['release_date']]
        song_data['year'] = [int(results['album']['release_date'].split('-')[0])]
        song_data['name'] = [name]
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]

        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)

    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

    def get_song_data(song):
        try:
            song_data=Music.objects(name=song['name']).first()
            if song_data==None:
                raise Exception()
            song_data_dict = {
                'name': song_data['name'],
                'artists': song_data['artists'],
                'explicit': song_data['explicit'],
                'duration_ms': song_data['duration_ms'],
                'popularity': song_data['popularity'],
                'valence': song_data['valence'],
                'year': song_data['year'],
                'acousticness': song_data['acousticness'],
                'danceability': song_data['danceability'],
                'duration_ms': song_data['duration_ms'],
                'energy': song_data['energy'],
                'instrumentalness': song_data['instrumentalness'],
                'key': song_data['key'],
                'liveness': song_data['liveness'],
                'loudness': song_data['loudness'],
                'mode': song_data['mode'],
                'popularity': song_data['popularity'],
                'speechiness': song_data['speechiness'],
                'tempo': song_data['tempo']
            }

            return pd.DataFrame([song_data_dict])

        except Exception as e:
            song_data=find_song(song['name'])
            if song_data is None or song_data.empty:
                return song_data
            artists_list=song_data['artists'].tolist()
            song_data_dict=song_data.to_dict()
            save_song_data=Music(
                name=song_data_dict['name'][0],
                explicit=song_data_dict['explicit'][0],
                duration_ms=song_data_dict['duration_ms'][0],
                popularity=song_data_dict['popularity'][0],
                danceability=song_data_dict['danceability'][0],
                energy=song_data_dict['energy'][0],
                key=song_data_dict['key'][0],
                loudness=song_data_dict['loudness'][0],
                mode=song_data_dict['mode'][0],
                speechiness=song_data_dict['speechiness'][0],
                acousticness=song_data_dict['acousticness'][0],
                instrumentalness=song_data_dict['instrumentalness'][0],
                liveness=song_data_dict['liveness'][0],
                valence=song_data_dict['valence'][0],
                tempo=song_data_dict['tempo'][0],
                id=song_data_dict['id'][0],
                release_date=song_data_dict['release_date'][0],
                year=song_data_dict['year'][0],
                artists=artists_list
            )
            save_song_data.save()
            return song_data

    def get_mean_vector(song_list):

        song_vectors = []

        for song in song_list:
            song_data = get_song_data(song)
            if song_data is None:
                print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
                continue
            song_vector = song_data[number_cols].values
            song_vectors.append(song_vector)
        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix, axis=0)

    def flatten_dict_list(dict_list):

        flattened_dict = defaultdict()
        for key in dict_list[0].keys():
            flattened_dict[key] = []

        for dictionary in dict_list:
            for key, value in dictionary.items():
                flattened_dict[key].append(value)

        return flattened_dict

    def recommend_songs(song_list, spotify_data, n_songs=10):

        metadata_cols = ['name', 'artists', 'id']
        song_dict = flatten_dict_list(song_list)

        song_center = get_mean_vector(song_list)
        scaler = song_cluster_pipeline.steps[0][1]
        scaled_data = scaler.transform(spotify_data[number_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1, -1))
        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])

        rec_songs = spotify_data.iloc[index]
        rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
        return rec_songs[metadata_cols].to_dict(orient='records')

    # song_list = [{'name': 'TROUBLESHOOTER'},
    #              {'name': 'AD MARE'},
    #              {'name': 'DOUBLAST'},
    #              {'name': 'Like'},
    #              {'name': 'Euphoria'}]

    res = recommend_songs(song_list, data)

    return res
