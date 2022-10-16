from django.contrib import admin

from .models import Article, UserHistory


class ArticleAdmin(admin.ModelAdmin):
    model = Article
    list_display = [
        "article_id", "title", "abstract", "category", "url", "timestamp",
    ]


class UserHistoryAdmin(admin.ModelAdmin):
    model = UserHistory
    list_display = [
        "id", "user_id", "article_id"
    ]


admin.site.register(Article, ArticleAdmin)
admin.site.register(UserHistory, UserHistoryAdmin)
