from django.contrib import admin

from .models import Nodes, Feeds, CropImage

# Register your models here.

admin.site.register(Nodes)
admin.site.register(Feeds)
admin.site.register(CropImage)
