from django.db import models
from mongoengine import Document, StringField, IntField, FloatField, ListField

# Create your models here.

class Music(Document):
    _id = StringField(primary_key=True, null=False, max_length=255, unique=True)
    valence = FloatField(null=True)
    year= IntField(null=True)
    acousticness = FloatField(null=True)
    artists= ListField(StringField(max_length=255), null=True)
    danceability = FloatField(null=True)
    duration_ms = IntField(null=True)
    energy = FloatField(null=True)
    explicit = IntField(null=True)
    id = StringField(null=False, max_length=255, unique=True)
    instrumentalness = FloatField(null=True)
    key = FloatField(null=True)
    liveness = FloatField(null=True)
    loudness = FloatField(null=True)
    mode = IntField(null=True)
    name= StringField(max_length=255, null=False)
    popularity = IntField(null=True)
    release_date = StringField(max_length=255, null=True)
    speechiness = FloatField(null=True)
    tempo = FloatField(null=True)
    class Meta:
        db_table = u'Music'




