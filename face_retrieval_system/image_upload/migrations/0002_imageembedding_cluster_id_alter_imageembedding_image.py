
# Generated by Django 5.2.3 on 2025-06-19 08:16

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_upload', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='imageembedding',
            name='cluster_id',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='imageembedding',
            name='image',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='embeddings', to='image_upload.uploadedimage'),
        ),
    ]
