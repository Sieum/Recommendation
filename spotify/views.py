import pandas as pd
from rest_framework import status

from spotify import spotify_crawl
from .models import Music
from .serializers import MusicSerializer
import pandas as pd

pd.set_option('display.max_columns', None)  ## 모든 열을 출력한다.


from rest_framework.decorators import api_view
from rest_framework.response import Response
import rest_framework.status as http


@api_view(['GET'])
def genre(request):
    
    if request.method == 'GET':
        # genre_list = request.GET.getlist('genres', [])
        genre_list = ["k-pop", "k-pop girl group", "k-pop boy group"]
        recommended_music = spotify_crawl.recommend_by_genre(genre_list)
        for music in recommended_music:
            print(f"음악 제목: {music['name']}, 아티스트: {music['artists']}")
        recommended_music_ids = []
        for music in recommended_music:
            recommended_music_ids.append({music['id']})
            print({music['id']})
        return Response(recommended_music_ids, status=status.HTTP_200_OK)
