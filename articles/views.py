from django.views.generic import DetailView

from .models import Article


class ArticleView(DetailView):
    model = Article
    template_name = 'articles/article.html'
    context_object_name = "article"
