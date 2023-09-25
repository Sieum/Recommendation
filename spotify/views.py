# import spotify_crawl as sc

import pandas as pd

from spotify import spotify_crawl
from .models import Music
from .serializers import MusicSerializer
import pandas as pd
from django.forms.models import model_to_dict

pd.set_option('display.max_columns', None)  ## 모든 열을 출력한다.
# df = pd.read_csv('./data.csv')

from rest_framework.decorators import api_view
from rest_framework.response import Response
import rest_framework.status as http


@api_view(['GET'])
def music(request):
    # data = pd.read_csv("C:\\Users\\SSAFY\\Desktop\\data.csv")
    # print(data)
    #
    # recommend_list=recommend.recommend_songs([{'name': 'Come As You Are'},
    #                                           {'name': 'Smells Like Teen Spirit'},
    #                                           {'name': 'Lithium'},
    #                                           {'name': 'All Apologies'},
    #                                           {'name': 'Stay Away'}],data)
    # print(recommend_list)
    track_features = spotify_crawl.recommend_music();
    print(track_features)
#       track_features = spotify_crawl.music_crawl()

    # # for track_feature in track_features:
    # #     if track_feature[0]['speechiness'] >= 0.66:
    # #         continue
    # #     else:
    # #         print(track_feature[0]['speechiness'])
    # Music.objects.filter()
    # for track_feature in track_features:
    #     if track_feature[0]['speechiness'] >= 0.66:
    #         continue
    #     else:
    #         serializer = MusicSerializer(data=track_feature[0])
    #         if serializer.is_valid(raise_exception=True):
    #             serializer.save()
    return Response(status=http.HTTP_200_OK)

@api_view(['GET'])
def crawling(request):
    music_info_list = spotify_crawl.music_crawl()
    for music_info in music_info_list:
        music_dict = model_to_dict(music_info)
        serializer = MusicSerializer(data=music_dict)
        if serializer.is_valid():
            try:
                serializer.save()
            except:
                print(music_dict)
        else:
            print(serializer.errors)
    return Response(status=http.HTTP_200_OK)

@api_view(['GET'])
def genre_music(request):
    genre_list = ['k-pop', 'k-pop girl group', 'k-pop boy group']
    music_list = spotify_crawl.recommend_by_genre(genre_list)
    print(music_list)
    for music in music_list:
        print(f"음악 제목: {music['name']}, 아티스트: {music['artists']}")
    return Response(status=http.HTTP_200_OK)