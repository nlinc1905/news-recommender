import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from prometheus_fastapi_instrumentator import Instrumentator
from scipy.stats import beta

from .samplers import epsilon_greedy, thompson_sampling, UCB1


SAMPLING_ALGORITHMS = {
    "e_greedy": epsilon_greedy,
    "thompson": thompson_sampling,
    "ucb": UCB1,
}
EPSILON = os.environ.get("EPSILON", default=0.1)

# store data for all AB test campaigns in memory - it will have a small footprint
# it would be good to store this elsewhere though in case the API goes down
campaign_to_variant_map = {
    "Test Home Page": {
        "A": {
            "impressions": 10,
            "conversions": 5,
        },
        "B": {
            "impressions": 35,
            "conversions": 20,
        },
    }
}
user_to_campaign_variant_map = {
    "user1": {
        "assigned_campaign_variants": {
            # "Test Home Page": "A",
        }
    }
}


class NewCampaign(BaseModel):
    campaign: str
    variants: List[str]

    class Config:
        # this will be used as the example in Swagger docs
        schema_extra = {
            "example": {
                "campaign": "Test Home Page",
                "variants": ["A", "B"],
            }
        }


class VariantRequest(BaseModel):
    campaign: str
    user_id: str
    sampling_method: Optional[str]

    class Config:
        # this will be used as the example in Swagger docs
        schema_extra = {
            "example": {
                "campaign": "Test Home Page",
                "user_id": "user1",
            }
        }


class VariantResponse(BaseModel):
    variant_id: str


class Impression(BaseModel):
    campaign: str
    variant: str

    class Config:
        # this will be used as the example in Swagger docs
        schema_extra = {
            "example": {
                "campaign": "Test Home Page",
                "variant": "A",
            }
        }


class Conversion(BaseModel):
    campaign: str
    variant: str

    class Config:
        # this will be used as the example in Swagger docs
        schema_extra = {
            "example": {
                "campaign": "Test Home Page",
                "variant": "A",
            }
        }


app = FastAPI()
# Instrumentator().instrument(app).expose(app)


# start app with: uvicorn main:app --reload
# to view the app: http://localhost:8000
# to view metrics: http://localhost:8000/metrics
# when running with Prometheus, configure a job for this app in prometheus.yml and test at http://localhost:9090/targets


@app.post("/new_campaign")
def create_campaign(data: NewCampaign) -> dict:
    # overwrite if already exists
    campaign_to_variant_map[data.campaign] = {}
    for v in data.variants:
        campaign_to_variant_map[data.campaign].update({v: {}})
        campaign_to_variant_map[data.campaign][v].update({"impressions": 1, "conversions": 1})
    return campaign_to_variant_map[data.campaign]


@app.get("/check_user")
def check_if_user_has_been_assigned_a_variant(user_id: str) -> str:
    return str(user_id in user_to_campaign_variant_map)


@app.post("/new_user")
def create_user(user_id: str) -> dict:
    # overwrite if already exists
    user_to_campaign_variant_map[user_id] = {"assigned_campaign_variants": {}}
    return user_to_campaign_variant_map[user_id]


@app.post("/variant")
def sample_variant_from_posterior(data: VariantRequest) -> VariantResponse:
    if data.user_id not in list(user_to_campaign_variant_map):
        raise KeyError(f"No user with the ID {data.user_id} exists.")
    variant_id = user_to_campaign_variant_map\
        .get(data.user_id, '')\
        .get("assigned_campaign_variants", '')\
        .get(data.campaign, '')
    # if the user exists and has already been assigned to a variant
    if variant_id != '':
        return VariantResponse(variant_id=variant_id)
    else:
        # sample using the given method, default to thompson if key error
        sampler = SAMPLING_ALGORITHMS.get(data.sampling_method, thompson_sampling)
        variant_id = sampler(variant_vals=campaign_to_variant_map[data.campaign], eps=EPSILON)
        # store the user's assigned variant
        user_to_campaign_variant_map[data.user_id]['assigned_campaign_variants'][data.campaign] = variant_id
        return VariantResponse(variant_id=variant_id)


@app.post("/impression")
def register_impression(data: Impression):
    if data.campaign not in campaign_to_variant_map.keys():
        raise KeyError(f"The campaign {data.campaign} does not exist.")
    campaign_to_variant_map[data.campaign][data.variant]["impressions"] += 1
    return campaign_to_variant_map[data.campaign][data.variant]["impressions"]


@app.post("/conversion")
def register_conversion(data: Conversion):
    if data.campaign not in campaign_to_variant_map.keys():
        raise KeyError(f"The campaign {data.campaign} does not exist.")
    if (
        campaign_to_variant_map[data.campaign][data.variant]["conversions"] + 1
        >
        campaign_to_variant_map[data.campaign][data.variant]["impressions"]
    ):
        raise ValueError(f"Cannot have more conversions than impressions.")
    campaign_to_variant_map[data.campaign][data.variant]["conversions"] += 1
    return campaign_to_variant_map[data.campaign][data.variant]["conversions"]


@app.get("/stats")
def get_campaign_stats(campaign: str) -> dict:
    if campaign not in campaign_to_variant_map.keys():
        raise KeyError(f"The campaign {campaign} does not exist.")
    traces = []
    for v, v_data in campaign_to_variant_map[campaign].items():
        successes = max(v_data['conversions'], 1)
        failures = max(v_data['impressions'] - v_data['conversions'], 1)
        # simulate samples from a beta distribution with alpha = successes and beta = failures
        traces.append(beta.rvs(successes, failures, size=10000))
    # assume A/B test and that the first item, A, is the control
    # A/B/C tests are not yet implemented, so do not calculate delta
    if len(traces) == 2:
        traces.append(traces[0] - traces[1])  # delta
        traces.append((sum(traces[1]) - sum(traces[0])) / sum(traces[1]))  # lift of B over A
    else:
        traces.append([0])
        traces.append(0.)
    return {
        "Campaign": campaign,
        "Posterior Mean for Variant A": np.mean(traces[0]),
        "Posterior Mean for Variant B": np.mean(traces[1]),
        "Probability that A is worse than B": np.mean(traces[2] < 0),
        "Probability that A is better than B": np.mean(traces[2] > 0),
        "Lift of B": traces[3],
        "Variant Impressions": {k: v["impressions"] for k, v in campaign_to_variant_map[campaign].items()},
    }
