from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm

from .models import CustomUser


class CustomUserCreationForm(UserCreationForm):

    class Meta(UserCreationForm):
        """
        This class overrides the default fields
        for UserCreationForm
        """
        model = CustomUser
        fields = ('username', 'email')


class CustomUserChangeForm(UserChangeForm):

    class Meta:
        """
        This class overrides the default fields
        for UserChangeForm
        """
        model = CustomUser
        fields = ('email',)


class DeleteUserForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = []  # this is required, even if blank
