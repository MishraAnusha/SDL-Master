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


from django.db import models

class Feeds(models.Model):
    mvp = models.BooleanField()
    mvs = models.BooleanField()
    svp = models.BooleanField()
    svs = models.BooleanField()
    ro_1 = models.BooleanField()
    ro_2 = models.BooleanField()


