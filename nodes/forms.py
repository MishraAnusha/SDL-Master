from django import forms

from .models import Nodes, CropImage


class RegisterForm(forms.ModelForm):
    name = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.TextInput(
            attrs={'placeholder': 'Name', 'class': 'form-control'}
        )
    )
    description = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.Textarea(
            attrs={'placeholder': 'add description', 'class': 'form-control', 'rows': 5}
        )
    )
    status = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.RadioSelect(
            choices=((True, 'Active'), (False, 'Inactive')),
            attrs={'class': 'form-check-input'}
        )
    )
    thing_speak_fetch = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.RadioSelect(
            choices=((True, 'Yes'), (False, 'No')),
            attrs={'class': 'form-check-input'}
        )
    )
    latitude = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.TextInput(
            attrs={'placeholder': 'latitude', 'class': 'form-control'}
        )
    )
    longitude = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.TextInput(
            attrs={'placeholder': 'longitude', 'class': 'form-control'}
        )
    )
    user_api_key = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(
            attrs={'placeholder': 'user_api_key', 'class': 'form-control', 'id': 'user_api_key'}
        )
    )
    node_api_key = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(
            attrs={'placeholder': 'node_api_key', 'class': 'form-control', 'id': 'node_api_key'}
        )
    )
    channel_id = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(
            attrs={'placeholder': 'channel_id', 'class': 'form-control', 'id': 'channel_id'}
        )
    )

    class Meta:
        model = Nodes
        fields = ['name', 'status', 'description', 'latitude', 'longitude', 'thing_speak_fetch', 'user_api_key',
                  'node_api_key', 'channel_id']


class ImageUploadForm(forms.ModelForm):
    image = forms.ImageField(
        required=True,
        widget=forms.FileInput(
            attrs={'placeholder': 'image', 'class': 'form-control-file'}
        )
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 5, 'placeholder': 'add description'}),
        required=False,
    )

    class Meta:
        model = CropImage
        fields = ['image', 'description']

class CSVImportForm(forms.Form):
    csv_file = forms.FileField(
        required=True,
    )
