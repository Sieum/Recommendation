
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Music",
            fields=[
                (
                    "id",
                    models.CharField(max_length=255, primary_key=True, serialize=False),
                ),
                ("acousticness", models.FloatField(null=True)),
                ("analysis_url", models.CharField(max_length=255, null=True)),
                ("danceability", models.FloatField(null=True)),
                ("duration_ms", models.IntegerField(null=True)),
                ("energy", models.FloatField(null=True)),
                ("instrumentalness", models.FloatField(null=True)),
                ("key", models.FloatField(null=True)),
                ("liveness", models.FloatField(null=True)),
                ("loudness", models.FloatField(null=True)),
                ("mode", models.IntegerField(null=True)),
                ("speechiness", models.FloatField(null=True)),
                ("tempo", models.FloatField(null=True)),
                ("time_signature", models.IntegerField(null=True)),
                ("track_href", models.CharField(max_length=2000, null=True)),
                ("type", models.CharField(max_length=255, null=True)),
                ("uri", models.CharField(max_length=255, null=True)),
                ("valence", models.FloatField(null=True)),
            ],
            options={
                "db_table": "Music",
            },
        ),
    ]
