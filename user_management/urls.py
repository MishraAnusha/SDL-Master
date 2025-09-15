from django.urls import re_path
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', include('users.urls')),

    path("nodes/", include('nodes.urls')),

    re_path(r'^oauth/', include('social_django.urls', namespace='social')),
]
