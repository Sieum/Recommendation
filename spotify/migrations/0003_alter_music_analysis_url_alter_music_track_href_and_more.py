# Generated by Django 4.1 on 2023-09-06 05:37

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("spotify", "0002_alter_music_id"),
    ]

    operations = [
        migrations.AlterField(
            model_name="music",
            name="analysis_url",
            field=models.CharField(max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name="music",
            name="track_href",
            field=models.CharField(max_length=2000, null=True),
        ),
        migrations.AlterField(
            model_name="music",
            name="type",
            field=models.CharField(max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name="music",
            name="uri",
            field=models.CharField(max_length=255, null=True),
        ),
    ]
