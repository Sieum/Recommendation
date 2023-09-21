import spotipy ## 스포티파이
from spotipy.oauth2 import SpotifyClientCredentials
import pprint ## 판다스 보기 좋게 프린트
import pandas as pd ## 판다스
from crawling.settings import DATA_DIR ## 데이터셋 경로
pd.set_option('display.max_columns', None)  ## 판다스 모든 열 출력
from decouple import config ## .env 파일 읽기
from sklearn.preprocessing import MinMaxScaler ## 사이킷런 데이터 정규화
import os
import warnings
warnings.filterwarnings("ignore")

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


## 데이터
data_path = os.path.join(DATA_DIR, 'data.csv')
genre_path = os.path.join(DATA_DIR, 'data_by_genres.csv')
year_path = os.path.join(DATA_DIR, 'data_by_year.csv')

data = pd.read_csv(data_path)
genre_data = pd.read_csv(genre_path)
year_data = pd.read_csv(year_path)

## 스포티파이 개발자 로그인
cid = config('SPOTIFY_ID')
secret = config('SPOTYFY_SECRET')
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

## KMeans 군집화
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                      ('kmeans', KMeans(n_clusters=20,
                                                        verbose=False))
                                      ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)

def music_crawl():
    # 2021 노래 검색
    artist_name = []
    track_name = []
    artist_id = []
    track_id = []
    year = []
    genre = []
    track_features = []
    for i in range(0, 1, ):
        track_results = sp.search(q='year:2023', type='track', limit=5, offset=i)
        for i, t in enumerate(track_results['tracks']['items']):
            artist_name.append(t['artists'][0]['name'])
            artist_id.append(t['artists'][0]['id'])
            track_name.append(t['name'])
            track_id.append(t['id'])
            year.append(t['album']['release_date'].split('-')[0])

    track_df = pd.DataFrame({'artist_name': artist_name, 'track_name': track_name, 'track_id': track_id,
                             'artist_id': artist_id, 'year': year})

    acousticness = []
    danceability = []
    energy = []
    instrumentalness = []
    key = []
    liveness = []
    loudness = []
    mode = []
    tempo = []
    valence = []
    speechiness = []

    for t_id in track_df['track_id']:
        af = sp.audio_features(t_id)
        track_features.append(af)

        acousticness.append(af[0]['acousticness'])
        danceability.append(af[0]['danceability'])
        energy.append(af[0]['energy'])
        instrumentalness.append(af[0]['instrumentalness'])
        key.append(af[0]['key'])
        liveness.append(af[0]['liveness'])
        loudness.append(af[0]['loudness'])
        mode.append(af[0]['mode'])
        tempo.append(af[0]['tempo'])
        valence.append(af[0]['valence'])
        speechiness.append(af[0]['speechiness'])

    tf_df = pd.DataFrame({'track_id': track_id, 'track_name': track_name, 'artist_id': artist_id, 'artist_name': artist_name,
                          'year': year, 'acousticness': acousticness, 'danceability': danceability, 'energy': energy,
                          'instrumentalness': instrumentalness, 'key': key, 'liveness': liveness, 'loudness': loudness,
                          'mode': mode, 'tempo': tempo, 'valence': valence, 'speechiness': speechiness})

    # tf_df.drop(tf_df.index, inplace=True)

    # for item in track_features:
    #     for feat in item:
    #         tf_df = tf_df.append(feat, ignore_index=True)

    # scaler = MinMaxScaler()
    # scaler.fit(tf_df)
    # tf_scaled = scaler.transform(tf_df)
    # df_tf_scaled = pd.DataFrame(data=tf_scaled,
    #                             columns=['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
    #                                      'liveness', 'loudness', 'mode', 'tempo',
    #                                      'time_signature', 'valence', 'speechiness'])
    # print(tf_df)
    return track_features


def recommend_music():

    def find_song(name):
        song_data = defaultdict()
        results = sp.search(q='track: {}'.format(name), limit=1)
        if results['tracks']['items'] == []:
            return None

        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = sp.audio_features(track_id)[0]

        song_data['name'] = [name]
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]

        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)

    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

    def get_song_data(song, spotify_data):

        try:
            data=Music.objects(name=song['name']).first()
            song_data = spotify_data[(spotify_data['name'] == song['name'])].iloc[0]

            return song_data

        except IndexError:
            return find_song(song['name'])

    def get_mean_vector(song_list, spotify_data):

        song_vectors = []

        for song in song_list:
            song_data = get_song_data(song, spotify_data)
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

        metadata_cols = ['name', 'artists']
        song_dict = flatten_dict_list(song_list)

        song_center = get_mean_vector(song_list, spotify_data)
        scaler = song_cluster_pipeline.steps[0][1]
        scaled_data = scaler.transform(spotify_data[number_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1, -1))
        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])

        rec_songs = spotify_data.iloc[index]
        rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
        return rec_songs[metadata_cols].to_dict(orient='records')

    song_list = [{'name': 'Spring Day'},
                 {'name': 'Dynamite'},
                 {'name': 'Boy in Luv'},
                 {'name': 'Like'},
                 {'name': 'Euphoria'}]

    music_list = recommend_songs(song_list, data)

    return music_list

