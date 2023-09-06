# import spotify_crawl as sc

import pandas as pd

from spotify import spotify_crawl

pd.set_option('display.max_columns', None)  ## 모든 열을 출력한다.
# df = pd.read_csv('./data.csv')

from rest_framework.decorators import api_view
from rest_framework.response import Response
import rest_framework.status as http


@api_view(['GET'])
def music(request):
    spotify_crawl.music_crawl()
    return Response(status=http.HTTP_200_OK)
