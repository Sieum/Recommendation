from django.db import models

# Create your models here.
#
class Music(models.Model):
    id = models.TextField(primary_key=True, null=False)
    acousticness = models.FloatField(null=True)
    analysis_url = models.TextField(null=True)
    danceability = models.FloatField(null=True)
    duration_ms = models.IntegerField(null=True)
    energy = models.FloatField(null=True)
    instrumentalness = models.FloatField(null=True)
    key = models.FloatField(null=True)
    liveness = models.FloatField(null=True)
    loudness = models.FloatField(null=True)
    mode = models.IntegerField(null=True)
    speechiness = models.FloatField(null=True)
    tempo = models.FloatField(null=True)
    time_signature = models.IntegerField(null=True)
    track_href = models.TextField(null=True)
    type = models.TextField(null=True)
    uri = models.TextField(null=True)
    valence = models.FloatField(null=True)

