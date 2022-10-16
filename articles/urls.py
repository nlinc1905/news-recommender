from django.urls import path

from .views import ArticleView


urlpatterns = [
    path('<int:pk>/', ArticleView.as_view(), name='article'),
]
