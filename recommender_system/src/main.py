import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator
import slowapi
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from recommenders.models.newsrec.io.mind_iterator import MINDIterator

from .models import RecommendersNewsrecModel
from .utils import add_noise_for_exploration


SEED = os.environ.get("SEED", default=14)
EPOCHS = os.environ.get("EPOCHS", default=1)
BATCH_SIZE = os.environ.get("BATCH_SIZE", default=64)
MODEL_TYPE = os.environ.get("MODEL_TYPE", default="nrms")
MODELS_PATH = Path(os.environ.get("MODELS_PATH", default="models/"))


class BehaviorModel(BaseModel):
    user_id: int
    article_id: int


class ArticleModel(BaseModel):
    article_id: int


class UserArticle(BaseModel):
    behavior: List[BehaviorModel]
    articles: List[ArticleModel]
    model: str
    explore_ratio: float


class UserRecommendationRequest(BaseModel):
    data: List[UserArticle]

    class Config:
        # this will be used as the example in Swagger docs
        schema_extra = {
            "example": {
                "data": [{
                        "behavior": [
                            {
                                "user_id": 4999,
                                "article_id": 25683
                            },
                            {
                                "user_id": 4999,
                                "article_id": 15844
                            },
                            {
                                "user_id": 4999,
                                "article_id": 13
                            }
                        ],
                        "articles": [
                            {
                                "article_id": 54
                            },
                            {
                                "article_id": 107
                            },
                            {
                                "article_id": 98
                            },
                            {
                                "article_id": 77
                            },
                            {
                                "article_id": 43
                            },
                        ],
                        "model": "nrms",
                        "explore_ratio": 0.0
                }]
            }
        }


class PredictionResponse(BaseModel):
    data: dict


if not os.path.exists(f"{MODELS_PATH}/{MODEL_TYPE}/ckpt.index"):
    m = RecommendersNewsrecModel(
        iterator=MINDIterator,
        model_type=MODEL_TYPE,
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    m.train()
    # m.eval()
    m.save(models_path=MODELS_PATH)
    m.load(models_path=MODELS_PATH)
    m.prepare_test_data()


limiter = slowapi.Limiter(key_func=get_remote_address)
app = FastAPI()
# Instrumentator().instrument(app).expose(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, slowapi._rate_limit_exceeded_handler)


# start app with: uvicorn main:app --reload
# to view the app: http://localhost:8000
# to view metrics: http://localhost:8000/metrics
# when running with Prometheus, configure a job for this app in prometheus.yml and test at http://localhost:9090/targets


@app.post("/")
@limiter.limit("1/second")
def recommend(data: UserRecommendationRequest, request: Request) -> PredictionResponse:
    request_dict = data.dict()['data'][0]
    # Reload the model here to make sure it's in the same thread
    # see: https://stackoverflow.com/questions/51127344/tensor-is-not-an-element-of-this-graph-deploying-keras-model
    # also: https://stackoverflow.com/questions/64221720/fastapi-and-python-threads
    m = RecommendersNewsrecModel(
        iterator=MINDIterator,
        model_type=MODEL_TYPE,
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    m.load(models_path=MODELS_PATH)
    resp = m.predict(request_dict)
    resp = add_noise_for_exploration(explore_ratio=request_dict['explore_ratio'], article_ranks=resp)
    return PredictionResponse(data=resp)
