import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pprint
import pandas as pd
pd.set_option('display.max_columns', None)  ## 모든 열을 출력한다.
# df = pd.read_csv('./data.csv')
from decouple import config

cid = config('SPOTIFY_ID')
secret = config('SPOTYFY_SECRET')
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def music_crawl():
    # 2021 노래 검색
    artist_name = []
    track_name = []
    track_popularity = []
    artist_id = []
    track_id = []
    track_features = []
    for i in range(0, 1, ):
        track_results = sp.search(q='year:2021', type='track', limit=50, offset=i)
        for i, t in enumerate(track_results['tracks']['items']):
            artist_name.append(t['artists'][0]['name'])
            artist_id.append(t['artists'][0]['id'])
            track_name.append(t['name'])
            track_id.append(t['id'])
            track_popularity.append(t['popularity'])

    track_df = pd.DataFrame({'artist_name': artist_name, 'track_name': track_name, 'track_id': track_id,
                             'track_popularity': track_popularity, 'artist_id': artist_id})


    for t_id in track_df['track_id']:
        af = sp.audio_features(t_id)
        track_features.append(af)
    tf_df = pd.DataFrame(
        columns=['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                 'liveness', 'valence', 'tempo', 'type', 'id', 'url', 'track_href', 'analysis_url', 'duration_ms',
                 'time_signature'])
    for item in track_features:
        for feat in item:
            tf_df = tf_df.append(feat, ignore_index=True)
    return track_features