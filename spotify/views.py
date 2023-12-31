# import spotify_crawl as sc

import pandas as pd

from spotify import spotify_crawl
from .models import Music
from .serializers import MusicSerializer
import pandas as pd

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
