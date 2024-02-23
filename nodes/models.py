from datetime import datetime

from PIL import Image
from django.db import models
from django.utils.timezone import now


# Create your models here.


class Nodes(models.Model):
    name = models.CharField(max_length=100)
    user_id = models.IntegerField(default=0)
    status = models.BooleanField(default=False)
    description = models.TextField(default='')
    latitude = models.FloatField()
    longitude = models.FloatField()
    last_feed_time = models.DateTimeField(null=True)
    last_feed_entry = models.IntegerField(null=True)
    thing_speak_fetch = models.BooleanField(default=False)
    user_api_key = models.CharField(max_length=100, null=True)
    node_api_key = models.CharField(max_length=100, null=True)
    channel_id = models.IntegerField(null=True)
    created_at = models.DateTimeField(default=datetime.now)
    updated_at = models.DateTimeField(default=datetime.now)

    def __str__(self):
        return self.name


class Feeds(models.Model):
    node_id = models.IntegerField()
    entry_id = models.IntegerField(null=True)
    temperature = models.FloatField(null=True)
    humidity = models.FloatField(null=True)
    LWS = models.FloatField(null=True)
    soil_temperature = models.FloatField(null=True)
    soil_moisture = models.FloatField(null=True)
    battery_status = models.FloatField(null=True)
    MVP = models.IntegerField(null=True)
    MVS = models.IntegerField(null=True)
    SVP = models.IntegerField(null=True)
    SVS = models.IntegerField(null=True)
    RO_1 = models.IntegerField(null=True)
    RO_2 = models.IntegerField(null=True)
    GWC = models.FloatField(null=True)
    duration = models.FloatField(null=True, default=0)
    event = models.BooleanField(null=True, default=0)
    powdery_mildew = models.FloatField(null=True, default=0)
    anthracnose = models.FloatField(null=True, default=0)
    root_rot = models.FloatField(null=True, default=0)
    irrigation = models.FloatField(null=True, default=0)
    created_at = models.DateTimeField(default=now)

    def __str__(self):
        return str(self.node_id)


class CropImage(models.Model):
    node_id = models.IntegerField()
    image = models.ImageField(upload_to='crop_images')
    description = models.TextField(default='')
    created_at = models.DateTimeField(default=datetime.now)

    def __str__(self):
        return str(self.node_id)

    # resizing images
    def save(self, *args, **kwargs):
        super().save()

        img = Image.open(self.image.path)

        if img.height > 100 or img.width > 100:
            new_img = (100, 100)
            img.thumbnail(new_img)
            img.save(self.image.path)

class SensorData(models.Model):
    mvp = models.BooleanField()
    mvs = models.BooleanField()
    svp = models.BooleanField()
    svs = models.BooleanField()
    ro_1 = models.BooleanField()
    ro_2 = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)
