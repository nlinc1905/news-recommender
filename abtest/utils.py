import numpy as np
import random
import scipy.stats
import json
from scipy.special import betaln

from .models import Campaign, Variant


def ab_assign(
        request,
        campaign,
        default_template: str,
        sticky_session: bool = True,
        algorithm: str = 'thompson',
        egreedy_eps: float = 0.1
) -> dict:
    """
    Determines the HTML template to serve for a given request (which variant of the AB test).
    Code adapted from: https://royhung.com/bayesian-ab-testing

    :param request: request object from Django
    :param campaign: campaign data model object
    :param default_template: file path to the default template for the view that this function modifies
    :param sticky_session: if True, a user will see the same variant every time they visit the page, as Django will
        remember the session variables
    :param algorithm: the search algorithm used to handle the explore/exploit tradeoff.  Possible values are:
        {
            'thompson': Thompson sampling,
            'UCB1': Upper confidence bound sampling,
            'uniform': Uniform random sampling,
            'egreedy': Epsilon-greedy with exploration determined by the 'eps' argument
        }
    :param egreedy_eps: The exploration ratio used in the Epsilon greedy strategy.

    :return: dictionary of
        {
            'variant_code': 'A',
            'impressions': 10,
            'conversions': 5,
            'conversion_rate': 0.5,
            'html_template': 'abtest/home_variant_a.html'
        }
    """
    variants = campaign.variants.all().values(
        'variant_code',
        'impressions',
        'conversions',
        'conversion_rate',
        'html_template',
    )

    # Sticky sessions - User gets previously assigned template
    campaign_code = str(campaign.campaign_id)
    if request.session.get(campaign_code):
        if request.session.get(campaign_code).get('variant_code') and sticky_session:
            return request.session.get(campaign_code)
    else:
        # Register new session variable
        request.session[campaign_code] = {
            'impressions': 1,  # Session impressions
            'conversions': 0,  # Session conversions
        }

    if algorithm == 'egreedy':
        assigned_variant = epsilon_greedy(variants, eps=egreedy_eps)
    elif algorithm == 'UCB1':
        assigned_variant = UCB1(variants)
    elif algorithm == 'uniform':
        assigned_variant = random.sample(list(variants), 1)[0]
    else:
        assigned_variant = thompson_sampling(variants)

    # Record assigned template in session variable
    request.session[campaign_code] = {
        **request.session[campaign_code],
        **assigned_variant
    }
    request.session.modified = True

    return assigned_variant


def epsilon_greedy(variant_vals: List[dict], eps: float = 0.1) -> dict:
    """
    Epsilon-greedy algorithm implementation on Variant model values.
    Code adapted from: https://royhung.com/bayesian-ab-testing

    :param variant_vals: list of dictionaries of variant values for a given campaign.  The dictionary keys are:
        ['variant_code', 'impressions', 'conversions', 'conversion_rate', 'html_template']
    :param eps: Exploration ratio, between 0 and 1.

    :return: selected variant dictionary with mappings of the variant fields (the columns in variant_vals), modified
        by the search algorithm
    """
    # If random number < eps, exploration is chosen over exploitation
    if random.random() < eps:
        selected_variant = random.sample(list(variant_vals), 1)[0]
    # Otherwise exploit by choosing the variant with the highest conversion rate (greedy search)
    else:
        best_conversion_rate = 0.0
        selected_variant = None
        for v in variant_vals:
            if v['conversion_rate'] > best_conversion_rate:
                best_conversion_rate = v['conversion_rate']
                selected_variant = v
            # break ties by randomly choosing either the current selection or the best variant
            if v['conversion_rate'] == best_conversion_rate:
                selected_variant = random.sample([v, selected_variant], 1)[0]
    return selected_variant


def thompson_sampling(variant_vals: List[dict]) -> dict:
    """
    Thompson Sampling algorithm implementation on Variant model values.
    Code adapted from: https://royhung.com/bayesian-ab-testing

    :param variant_vals: list of dictionaries of variant values for a given campaign.  The dictionary keys are:
        ['variant_code', 'impressions', 'conversions', 'conversion_rate', 'html_template']

    :return: selected variant dictionary with mappings of the variant fields (the columns in variant_vals), modified
        by the search algorithm
    """
    selected_variant = None
    best_sample = 0.0
    for v in variant_vals:
        # sample from the beta distribution where
        # alpha = max(conversions, 1) and
        # beta = max((impressions - conversions), 1)
        sample = np.random.beta(
            max(v['conversions'], 1),
            max(v['impressions'] - v['conversions'], 1)
        )
        if sample > best_sample:
            best_sample = sample
            selected_variant = v
    return selected_variant


def UCB1(variant_vals: List[dict]) -> dict:
    """
    Upper Confidence Bound algorithm implementation on Variant model values.
    Code adapted from: https://royhung.com/bayesian-ab-testing

    :param variant_vals: list of dictionaries of variant values for a given campaign.  The dictionary keys are:
        ['variant_code', 'impressions', 'conversions', 'conversion_rate', 'html_template']

    :return: selected variant dictionary with mappings of the variant fields (the columns in variant_vals), modified
        by the search algorithm
    """
    selected_variant = None
    best_score = 0.0
    total_impressions = sum([var['impressions'] for var in variant_vals])
    for v in variant_vals:
        score = v['conversion_rate'] + np.sqrt(2 * np.log(total_impressions) / v['impressions'])
        if score > best_score:
            best_score = score
            selected_variant = v
        if score == best_score:
            # Tie breaker
            selected_variant = random.sample([v, selected_variant], 1)[0]
    return selected_variant


def h(a, b, c, d):
    """
    Closed form solution for P(X>Y).
    Where: X ~ Beta(a,b), Y ~ Beta(c,d)
    Code adapted from: https://royhung.com/bayesian-ab-testing

    Parameters
    ----------
    a : int
        alpha shape parameter for the beta distribution. a > 0
    b : int
        beta shape parameter for the beta distribution. b > 0

    Returns
    -------
    float
        Returns probability of X > Y

    References
    ----------
    https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html

    """
    total = 0.0
    for j in range(c):
        total += np.exp(betaln(a + j, b + d) - np.log(d + j) - betaln(1 + j, d) - betaln(a, b))
    return 1 - total


def loss(a, b, c, d):
    """
    Expected loss function built on P(X>Y)
    Where: X ~ Beta(a,b), Y ~ Beta(c,d)
    Code adapted from: https://royhung.com/bayesian-ab-testing

    Parameters
    ----------
    a : int
        alpha shape parameter for the beta distribution. a > 0
    b : int
        beta shape parameter for the beta distribution. b > 0

    Returns
    -------
    float
        Returns the expected loss in terms of conversion rate
        when you pick variant Y over X when variant X actually has a higher
        conversion rate than Y.

    References
    ----------
        https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html
        https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf

    """
    return np.exp(betaln(a + 1, b) - betaln(a, b)) * h(a + 1, b, c, d) - \
           np.exp(betaln(c + 1, d) - betaln(c, d)) * h(a, b, c + 1, d)


def sim_page_visits(campaign, n, conversion_rates, algo='thompson', eps=0.1, ):
    """
    Simulate `n` page visits to the page that is being A/B tested.
    The probability of each simulated page visited generating a conversion
    is determined by the conversion rates provided in the `conversion_rates` param.

    Parameters
    ----------
    campaign : :obj:`Campaign`
        The A/B test Campaign model object which will be subject to the simulation
    conversion_rates : dict: ``{variant_code: probability}``
        Dictionary mapping to contain the probability of conversion for each
        simulated page visit for each Variant available in the campaign.
        If key-value pair for variant not provided in the mapping, that
        variant's probability of conversion will default to 0.5.
    n : int
        Number of page visits to simulate.
    algo : str, optional
        Algorithm to determine the assignment (explore-exploit) of the Variant
        to the page request.
        Valid values are ``thompson``, ``egreedy``, ``uniform``, ``UCB1``,
        Defaults to ``thompson``.
    eps : float, optional
        Exploration parameter for the epsilon-greedy ``egreedy`` algorithm.
        Only applicable to ``egreedy`` algorithm option. Defaults to 0.1


    Returns
    -------
    bool
        True if successful
    """

    variants = campaign.variants.all().values(
        'variant_code',
        "impressions",
        'conversions',
        'conversion_rate',
        'html_template',
    )
    for i in range(n):
        if algo == 'thompson':
            assigned_variant = thompson_sampling(variants)
        if algo == 'egreedy':
            assigned_variant = epsilon_greedy(variants, eps=eps)
        if algo == 'UCB1':
            assigned_variant = UCB1(variants)
        if algo == 'uniform':
            assigned_variant = random.sample(list(variants), 1)[0]

        # Simulate user conversion after version assigned
        conversion_prob = conversion_rates.get(assigned_variant['variant_code'], 0.5)
        conversion = 1 if random.random() > 1 - conversion_prob else 0
        variant = Variant.objects.get(
            campaign=campaign,
            variant_code=assigned_variant['variant_code']
        )
        variant.impressions = variant.impressions + 1
        variant.conversions = variant.conversions + conversion
        variant.conversion_rate = variant.conversions / variant.impressions
        variant.save()

    return True
