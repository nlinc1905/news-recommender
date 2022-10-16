from django import forms

from .models import UserHistory


class UserArticleForm(forms.ModelForm):

    class Meta:
        model = UserHistory
        fields = ["user_id", "article_id"]
        widgets = {
            'user_id': forms.TextInput(attrs={
                'id': 'form_userid',
            }),
            'article_id': forms.TextInput(attrs={
               'id': 'form_articleid',
            }),
        }
