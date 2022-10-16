import os
import pandas as pd
import numpy as np
from pathlib import Path
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set


def download_mind_data(data_path: str = "data/", model_type: str = "nrms", mind_type: str = "demo"):
    """
    Downloads MIND data.  See starter notebook here:
    https://github.com/microsoft/recommenders/blob/main/examples/00_quick_start/nrms_MIND.ipynb

    :param data_path: where to save the data
    :param model_type: which type of model to build.
        Options come from: https://github.com/microsoft/recommenders/tree/main/recommenders/models/newsrec/models
    :param mind_type: demo, small, or large
    """
    if model_type not in ['lstur', 'naml', 'npa', 'nrms']:
        raise ValueError(f"model_type should be 1 of 'lstur', 'naml', 'npa', 'nrms', but received {model_type}")
    if mind_type not in ['demo', 'small', 'large']:
        raise ValueError(f"MIND_type should be 1 of 'demo', 'small', 'large', but received {mind_type}")

    # Create folders for the MIND data
    path = Path(data_path)
    path_train = path / "train"
    path_valid = path / "valid"
    path_utils = path / "utils"
    path_train.mkdir(exist_ok=True)
    path_valid.mkdir(exist_ok=True)
    path_utils.mkdir(exist_ok=True)
    file_paths = {
        "train_news": path_train / 'news.tsv',
        "train_behaviors": path_train / 'behaviors.tsv',
        "valid_news": path_valid / 'news.tsv',
        "valid_behaviors": path_valid / 'behaviors.tsv',
        "glove_embed": path_utils / 'embedding.npy',
        "user_dict": path_utils / 'uid2index.pkl',
        "word_dict": path_utils / 'word_dict.pkl',
        "yaml": path_utils / f'{model_type}.yaml',
    }

    # Download the MIND data
    mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(mind_type)

    if not os.path.exists(file_paths['train_news']):
        download_deeprec_resources(mind_url, str(path_train), mind_train_dataset)
    if not os.path.exists(file_paths['valid_news']):
        download_deeprec_resources(mind_url, str(path_valid), mind_dev_dataset)
    if not os.path.exists(file_paths['yaml']):
        # this downloads the yamls for all models in the newsrec category (lstur, naml, np, and nrms)
        download_deeprec_resources(
            r'https://recodatasets.z20.web.core.windows.net/newsrec/',
            str(path_utils),
            mind_utils
        )

    return file_paths


def convert_id_to_int(df: pd.DataFrame, id_col: str):
    """Converts IDs to integers for use as primary keys"""
    temp = df[id_col].drop_duplicates().reset_index(drop=True).reset_index(drop=False)
    id_to_new_uid_map = dict(zip(temp[id_col], temp['index']))
    df[id_col] = df[id_col].map(id_to_new_uid_map).astype(int)
    return df, id_to_new_uid_map


def add_noise_for_exploration(explore_ratio: float, article_ranks: dict):
    """Randomly swaps explore_ratio percent of ranks to balance explore/exploit tradeoff"""
    if explore_ratio < 0.0 or explore_ratio > 1.0:
        raise ValueError("explore_ratio must be >= 0 and <= 1")

    if explore_ratio == 0.0:
        return article_ranks

    np.random.seed(14)

    # create an index of the original ranks
    ranks = list(article_ranks.values())
    rank_index = [i for i in range(len(ranks))]
    nbr_trials = len(ranks)

    # randomly select some items to swap
    nbr_swaps = int(np.ceil((explore_ratio * nbr_trials) / 2.) * 2)
    # ensure that there will never be more swaps than trials
    while nbr_swaps > nbr_trials:
        nbr_swaps -= 2  # nbr_swaps will always be even, so it must be reduced by 2
    print(f"Swapping {nbr_swaps} ranks.")
    ranks_to_swap = list(np.random.choice(rank_index, size=nbr_swaps, replace=False))

    # pair the items to determine which ones to swap with which
    ranks_to_swap = [ranks_to_swap[n:n+2] for n in range(0, len(ranks_to_swap), 2)]

    def swap(l, pos1, pos2):
        l[pos1], l[pos2] = l[pos2], l[pos1]
        return l

    for r in ranks_to_swap:
        ranks = swap(ranks, *r)
    return {k: ranks[i] for i, (k, v) in enumerate(article_ranks.items())}
