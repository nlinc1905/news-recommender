from django.urls import path

from .views import SignupPageView, delete_user_view


urlpatterns = [
    path('signup/', SignupPageView.as_view(), name='signup'),
    path('delete/', delete_user_view, name='delete_user'),
]