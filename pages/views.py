import os
import json
import requests
import datetime
from datetime import datetime
from django.views.generic import TemplateView, ListView, FormView, View
from django.http import JsonResponse

from articles.models import TopStories, UserHistory, Article
from articles.forms import UserArticleForm


def send_to_recommender_api(user, articles, return_rank=True):
    """
    Posts to the recommender API and parses the response.
    Response looks like:  {54: 1, 43: 0.9, 107: 0.3, 98: 0.8, 77: 0.4}
    """
    # get user history
    userhist = list(UserHistory.objects.filter(user_id=user).values("user_id", "article_id"))
    if len(userhist) == 0:
        return {}

    # send request
    data = {
        "behavior": userhist,
        "articles": articles,
        "model": os.getenv("MODEL", default="nrms"),
        "explore_ratio": float(os.getenv("EXPLORE_RATIO", default=0.0))
    }
    data = {"data": [data]}
    # get recommendations from the model API using the URL from the bridge network
    # see: https://docs.docker.com/network/network-tutorial-standalone/#use-the-default-bridge-network
    resp = requests.post(
        'http://172.17.0.1:5000/',
        data=json.dumps(data)
    )
    resp_dict = json.loads(resp.text)['data']
    resp_dict = {int(k): v for k, v in resp_dict.items()}
    return resp_dict


class HomePageDisplay(ListView):
    template_name = 'home.html'
    model = TopStories
    context_object_name = "news_list"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = UserArticleForm()
        # get articles to be scored by the model
        todays_articles = Article.objects.filter(
            timestamp__date=datetime.strptime(os.environ["TODAYS_DATE"], '%Y-%m-%d')
        )

        # try to get recommendations, if TypeError caused by anonymous user (not logged in) return todays_articles
        default_resp = {a.article_id: idx for idx, a in enumerate(todays_articles)}
        try:
            # get article ranks from the model
            api_resp = send_to_recommender_api(
                user=self.request.user,
                articles=list(todays_articles.values('article_id')),
                return_rank=True
            )
            # sort the model response by ascending rank (descending score)
            api_resp = list(sorted(api_resp, key=api_resp.get, reverse=True)) if len(api_resp) > 0 else default_resp
        except TypeError:
            api_resp = default_resp

        # sort the queryset results according to the ranked model response
        todays_articles = [a for _, a in sorted(
            zip(api_resp, todays_articles)
        )][:int(os.environ["NBR_RECOMMENDATIONS"])]
        context['recommendations'] = todays_articles
        return context


class HomePageSave(FormView):
    template_name = "home.html"
    form_class = UserArticleForm
    model = UserHistory

    def get_success_url(self):
        return ""

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            # only save if the user has not clicked on this article before
            user = form.cleaned_data.get("user_id", "")
            article = form.cleaned_data.get("article_id", "")
            user_has_already_clicked = UserHistory.objects.all().filter(user_id=user, article_id=article)
            if len(user_has_already_clicked) == 0:
                form.save()
        return super().post(request, *args, **kwargs)


class HomePageView(View):

    def get(self, request, *args, **kwargs):
        view = HomePageDisplay.as_view()
        return view(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        view = HomePageSave.as_view()
        return view(request, *args, **kwargs)


class AboutPageView(TemplateView):
    template_name = 'about.html'
