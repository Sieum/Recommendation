from django.db import models

# Create your models here.


class Music(models.Model):
    id = models.CharField(primary_key=True, null=False, max_length=255)
    acousticness = models.FloatField(null=True)
    analysis_url = models.CharField(null=True, max_length=255)
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
    track_href = models.CharField(null=True, max_length=2000)
    type = models.CharField(null=True, max_length=255)
    uri = models.CharField(null=True, max_length=255)
    valence = models.FloatField(null=True)

    class Meta:
        db_table = u'Music'

