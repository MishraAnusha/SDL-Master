from PIL import Image
from django.contrib.auth.models import User
from django.db import models


# Extending User Model Using a One-To-One Link
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar = models.ImageField(default='default.jpg', upload_to='profile_images')
    bio = models.TextField()

    def __str__(self):
        return self.user.username

    # resizing images
    def save(self, *args, **kwargs):
        super().save()

        img = Image.open(self.avatar.path)

        if img.height > 100 or img.width > 100:
            new_img = (100, 100)
            img.thumbnail(new_img)
            img.save(self.avatar.path)
            
class Node(models.Model):
    node_id = models.CharField(max_length=100)
    mvp = models.CharField(max_length=100)
    mvs = models.CharField(max_length=100)
    svp = models.CharField(max_length=100)
    svs = models.CharField(max_length=100)
    ro_1 = models.BooleanField(default=False)
    ro_2 = models.BooleanField(default=False)




