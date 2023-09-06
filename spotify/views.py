from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
import rest_framework.status as http

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pprint
import pandas as pd
# from models import Music
# from serializers import MusicSerializer

cid = 'eead00e8318343b9ad35cb2ae145e047'
secret = '9987a67355c74bb1b8da7515782ae4a2'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


@api_view(['GET'])
def music(request):
    # 2021 노래 검색
    artist_name = []
    track_name = []
    track_popularity = []
    artist_id = []
    track_id = []
    for i in range(0, 1,):
        track_results = sp.search(q='year:2021', type='track', limit=50, offset=i)
        for i, t in enumerate(track_results['tracks']['items']):
            artist_name.append(t['artists'][0]['name'])
            artist_id.append(t['artists'][0]['id'])
            track_name.append(t['name'])
            track_id.append(t['id'])
            track_popularity.append(t['popularity'])

    track_df = pd.DataFrame({'artist_name': artist_name, 'track_name': track_name, 'track_id': track_id,
                             'track_popularity': track_popularity, 'artist_id': artist_id})

    # pprint.pprint(track_df.shape)
    # pprint.pprint(track_df.head())

    pprint.pprint(track_df)
    # serializer = MusicSerializer()
    return Response(status=http.HTTP_200_OK)
