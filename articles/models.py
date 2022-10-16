import os
from datetime import datetime
from django.db import models
from django.urls import reverse
from django.contrib.auth import get_user_model


class Article(models.Model):
    """Data model for news articles"""
    article_id = models.AutoField(primary_key=True)
    title = models.TextField(null=True, blank=True)
    abstract = models.TextField(null=True, blank=True)
    category = models.CharField(max_length=255, null=True, blank=True)
    url = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=False, null=True, blank=True)

    class Meta:
        # default ordering is lowest article_id first
        ordering = ['article_id']

    def __str__(self):
        return str(self.article_id)

    def get_absolute_url(self):
        return reverse('article', args=[str(self.article_id)])


class TopStories(models.Model):
    """
    Data model for top stories.  These could be calculated on the fly, but to speed up
    web page loading, they will be pre-computed.
    """
    date = models.DateTimeField(auto_now_add=False)
    article_id = models.ForeignKey(Article, on_delete=models.CASCADE)
    clicks = models.IntegerField(default=0)
    title = models.TextField(null=True, blank=True)
    abstract = models.TextField(null=True, blank=True)
    category = models.CharField(max_length=255, null=True, blank=True)
    url = models.TextField(null=True, blank=True)

    class Meta:
        # default ordering is most clicks on the newest date
        ordering = ['-date', '-clicks']

    @property
    def is_today(self):
        """
        Strips timezone information from self.date and compares it to the environment variable
        that determines which stories to show.
        """
        return self.date.replace(tzinfo=None) == datetime.strptime(os.environ["TODAYS_DATE"], '%Y-%m-%d')


class UserHistory(models.Model):
    """Tracks which articles users read"""
    user_id = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    article_id = models.ForeignKey(Article, on_delete=models.CASCADE)
