import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pprint
import pandas as pd
pd.set_option('display.max_columns', None)  ## 모든 열을 출력한다.
# df = pd.read_csv('./data.csv')
from decouple import config

from sklearn.preprocessing import MinMaxScaler

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

    acousticness = []
    danceability = []
    energy = []
    instrumentalness = []
    key = []
    liveness = []
    loudness = []
    mode = []
    tempo = []
    time_signature = []
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
        time_signature.append(af[0]['time_signature'])
        valence.append(af[0]['valence'])
        speechiness.append(af[0]['speechiness'])

    tf_df = pd.DataFrame({'acousticness': acousticness, 'danceability': danceability, 'energy': energy,
                          'instrumentalness': instrumentalness, 'key': key, 'liveness': liveness, 'loudness': loudness,
                          'mode': mode, 'tempo': tempo, 'time_signature': time_signature, 'valence': valence, 'speechiness': speechiness})

    # tf_df.drop(tf_df.index, inplace=True)

    # for item in track_features:
    #     for feat in item:
    #         tf_df = tf_df.append(feat, ignore_index=True)

    scaler = MinMaxScaler()
    scaler.fit(tf_df)
    tf_scaled = scaler.transform(tf_df)
    df_tf_scaled = pd.DataFrame(data=tf_scaled,
                                columns=['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
                                         'liveness', 'loudness', 'mode', 'tempo',
                                         'time_signature', 'valence', 'speechiness'])
    print(df_tf_scaled)
    return track_features
