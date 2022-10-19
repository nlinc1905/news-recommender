from django.shortcuts import render, redirect
from .utils import ab_assign, h, sim_page_visits
from .simulation import experiment
from .models import Campaign, Variant
import numpy as np
import json
import datetime


def dashboard(request):
    """
    Dashboard to show progress of AB test
    """
    campaign = Campaign.objects.get(name="Test Homepage")
    variant_vals = list(campaign.variants.all().order_by('variant_code').values(
        'variant_code',
        'impressions',
        'conversions',
        'conversion_rate',
        'html_template',
    ))
    x_vals = list(np.linspace(0, 1, 500))
    xy_vals = []
    max_y = 0
    N = 0  # Total number of page visits
    COLOUR_PALETTE = [
        '#66c2a5',
        '#fc8d62',
        '#8da0cb',
        '#e78ac3',
        '#a6d854',
    ]

    for i, variant in enumerate(campaign.variants.all().order_by('code')):
        y_vals = variant.beta_pdf(x_vals)
        variant_vals[i]['xy'] = list(zip(x_vals, y_vals))
        variant_vals[i]['color'] = COLOUR_PALETTE[i % len(COLOUR_PALETTE)]
        if max(y_vals) > max_y:
            max_y = max(y_vals)
        N += variant_vals[i]['impressions']

    # Calculate pairwise probability of variant X conversion rate
    # greater than variant Y conversion rate

    h_ab = h(
        variant_vals[0]['conversions'],
        variant_vals[0]['impressions'] - variant_vals[0]['conversions'],
        variant_vals[1]['conversions'],
        variant_vals[1]['impressions'] - variant_vals[1]['conversions']
    )
    h_ac = h(
        variant_vals[0]['conversions'],
        variant_vals[0]['impressions'] - variant_vals[0]['conversions'],
        variant_vals[2]['conversions'],
        variant_vals[2]['impressions'] - variant_vals[2]['conversions']
    )
    h_ba = h(
        variant_vals[1]['conversions'],
        variant_vals[1]['impressions'] - variant_vals[1]['conversions'],
        variant_vals[0]['conversions'],
        variant_vals[0]['impressions'] - variant_vals[0]['conversions']
    )
    h_bc = h(
        variant_vals[1]['conversions'],
        variant_vals[1]['impressions'] - variant_vals[1]['conversions'],
        variant_vals[2]['conversions'],
        variant_vals[2]['impressions'] - variant_vals[2]['conversions']
    )
    h_ca = h(
        variant_vals[2]['conversions'],
        variant_vals[2]['impressions'] - variant_vals[2]['conversions'],
        variant_vals[0]['conversions'],
        variant_vals[0]['impressions'] - variant_vals[0]['conversions']
    )
    h_cb = h(
        variant_vals[2]['conversions'],
        variant_vals[2]['impressions'] - variant_vals[2]['conversions'],
        variant_vals[1]['conversions'],
        variant_vals[1]['impressions'] - variant_vals[1]['conversions']
    )

    context = {
        'campaign': campaign,
        'variant_vals': variant_vals,
        'x_vals': json.dumps(x_vals),
        'max_y': max_y,
        'N': N,
        'h_ab': h_ab,
        'h_ac': h_ac,
        'h_ba': h_ba,
        'h_bc': h_bc,
        'h_ca': h_ca,
        'h_cb': h_cb,
        'last_update': datetime.datetime.utcnow().strftime('%Y-%m-%d | %H:%M:%S')
    }
    return render(
        request,
        'abtest/dashboard.html',
        context
    )


def clear_stats(request):
    """
    For demonstration purposes only.
    Clears all variant impressions / conversions
    """
    Variant.objects.all().update(
        conversions=0,
        impressions=0,
        conversion_rate=0.0,
    )
    return redirect(dashboard)


def simulation(request):

    dataset = experiment(
        p1=0.3,
        p2=0.6,
        p3=0.65,
        N=10000,
        algo="thompson",
    )
    context = {
        'dataset': json.dumps(dataset)
    }
    return render(
        request,
        'abtest/simulation.html',
        context
    )
