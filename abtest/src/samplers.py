import numpy as np
import random


def get_conversion_rate(imp: int, conv: int) -> float:
    return float(imp / conv)


def epsilon_greedy(variant_vals: dict, eps: float = 0.1) -> str:
    """
    Epsilon-greedy algorithm implementation on Variant model values.

    :param variant_vals: dictionary of: {"variant_id": {"impressions": int, "conversions": int}}
    :param eps: Exploration ratio, between 0 and 1.

    :return: selected variant_id (the key from variant_vals)
    """
    # If random number < eps, exploration is chosen over exploitation
    if random.random() < eps:
        selected_variant = random.sample(list(variant_vals), 1)[0]
    # Otherwise exploit by choosing the variant with the highest conversion rate (greedy search)
    else:
        best_conversion_rate = 0.0
        selected_variant = None
        for v, v_data in variant_vals.items():
            v_conv_rate = get_conversion_rate(imp=v_data['impressions'], conv=v_data['conversions'])
            if v_conv_rate > best_conversion_rate:
                best_conversion_rate = v_conv_rate
                selected_variant = v
            # break ties by randomly choosing either the current selection or the best variant
            if v_conv_rate == best_conversion_rate:
                selected_variant = random.sample([v, selected_variant], 1)[0]
    return selected_variant


def thompson_sampling(variant_vals: dict, **kwargs) -> str:
    """
    Thompson Sampling algorithm implementation on Variant model values.

    :param variant_vals: dictionary of: {"variant_id": {"impressions": int, "conversions": int}}

    :return: selected variant_id (the key from variant_vals)
    """
    selected_variant = None
    best_sample = 0.0
    for v, v_data in variant_vals.items():
        # sample from the beta distribution where
        # alpha = successes = max(conversions, 1) and
        # beta = failures = max((impressions - conversions), 1)
        sample = np.random.beta(
            max(v_data['conversions'], 1),
            max(v_data['impressions'] - v_data['conversions'], 1)
        )
        if sample > best_sample:
            best_sample = sample
            selected_variant = v
    return selected_variant


def UCB1(variant_vals: dict, **kwargs) -> str:
    """
    Upper Confidence Bound algorithm implementation on Variant model values.

    :param variant_vals: dictionary of: {"variant_id": {"impressions": int, "conversions": int}}

    :return: selected variant_id (the key from variant_vals)
    """
    selected_variant = None
    best_score = 0.0
    total_impressions = sum([var['impressions'] for var in variant_vals.values()])
    for v, v_data in variant_vals.items():
        v_conv_rate = get_conversion_rate(imp=v_data['impressions'], conv=v_data['conversions'])
        score = v_conv_rate + np.sqrt(2 * np.log(total_impressions) / v_data['impressions'])
        if score > best_score:
            best_score = score
            selected_variant = v
        if score == best_score:
            # break ties by randomly choosing either the current selection or the best variant
            selected_variant = random.sample([v, selected_variant], 1)[0]
    return selected_variant
