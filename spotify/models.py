from django.db import models
from mongoengine import Document, StringField, IntField, FloatField, ListField

# Create your models here.

class Music(models.Model):
    valence = models.FloatField(null=True)
    year= models.FloatField(null=True)
    acousticness = models.FloatField(null=True)
    artists= models.CharField(max_length=255, null=True)
    danceability = models.FloatField(null=True)
    duration_ms = models.FloatField(null=True)
    energy = models.FloatField(null=True)
    explicit = models.FloatField(null=True)
    id = models.CharField(primary_key=True, null=False, max_length=255, unique=True)
    instrumentalness = models.FloatField(null=True)
    key = models.FloatField(null=True)
    liveness = models.FloatField(null=True)
    loudness = models.FloatField(null=True)
    mode = models.FloatField(null=True)
    name = models.CharField(max_length=255, null=False)
    popularity = models.FloatField(null=True)
    release_date = models.CharField(max_length=255, null=True)
    speechiness = models.FloatField(null=True)
    tempo = models.FloatField(null=True)
    class Meta:
        db_table = u'Music'




