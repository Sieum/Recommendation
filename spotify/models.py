from django.db import models

# Create your models here.


class Music(models.Model):
    id = models.CharField(primary_key=True, null=False, max_length=255, unique=True)
    acousticness = models.FloatField(null=True)
    danceability = models.FloatField(null=True)
    energy = models.FloatField(null=True)
    instrumentalness = models.FloatField(null=True)
    key = models.FloatField(null=True)
    liveness = models.FloatField(null=True)
    loudness = models.FloatField(null=True)
    mode = models.IntegerField(null=True)
    tempo = models.FloatField(null=True)
    time_signature = models.IntegerField(null=True)
    valence = models.FloatField(null=True)

    class Meta:
        db_table = u'Music'

