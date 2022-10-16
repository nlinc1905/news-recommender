import os
import json
import pandas as pd
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


def get_user_article_pairs(behavior_df: pd.DataFrame):
    """
    Pairs each user with their article IDs.  These pairings will be used to form a user-item matrix.
    Articles that were not clicked on look like "ID-0" in the impressions column.  These will be treated
    as 0's in the user-item matrix.  In the future, it may be worth treating them differently, because
    the user presumably saw them but was uninterested in them.  All articles in the history will be treated
    as 1's in the user-item matrix.
    """
    # pair user with articles from impressions
    behavior_df["impressions"] = behavior_df["impressions"].apply(lambda s: str(s).split())
    long_form_df_imp = behavior_df.explode("impressions")
    long_form_df_imp = long_form_df_imp[["user_id", "impressions"]]
    long_form_df_imp[["article_id", "clicked"]] = long_form_df_imp["impressions"].str.split("-", expand=True)
    long_form_df_imp = long_form_df_imp[long_form_df_imp["clicked"] == "1"][["user_id", "article_id"]]

    # pair user with articles from history
    behavior_df["history"] = behavior_df["history"].apply(lambda s: str(s).split())
    long_form_df_his = behavior_df.explode("history")
    long_form_df_his = long_form_df_his[["user_id", "history"]].rename(columns={"history": "article_id"})
    long_form_df_his = long_form_df_his[["user_id", "article_id"]]

    # combine articles from impressions and history
    user_article_pairs = pd.concat([long_form_df_imp, long_form_df_his], axis=0).drop_duplicates()
    user_article_pairs = user_article_pairs.reset_index(drop=False).rename(columns={"index": "id"})

    return user_article_pairs


def get_impression_article_times(behavior_df: pd.DataFrame):
    """
    Get a timestamp for each impression article by finding the earliest impression times.
    """
    behavior_df["impressions"] = behavior_df["impressions"].apply(lambda s: str(s).split())
    long_form_df_imp = behavior_df.explode("impressions")
    long_form_df_imp = long_form_df_imp[["time", "impressions"]]
    long_form_df_imp["time"] = pd.to_datetime(long_form_df_imp["time"])
    long_form_df_imp = long_form_df_imp[["impressions", "time"]].rename(columns={"impressions": "article_id"})
    long_form_df_imp["article_id"] = long_form_df_imp["article_id"].apply(lambda s: s.split("-")[0])
    long_form_df_imp = long_form_df_imp.groupby("article_id")["time"].min().reset_index()
    return long_form_df_imp


def remove_missing_article_ids(df: pd.DataFrame, article_df: pd.DataFrame):
    """
    Ensures that there are no article_ids in the user-article pairs that do not exist in the article data.
    Ensures that there are no article_ids in the article timestamps that do not exist in the article data.
    """
    article_ids = set(article_df["article_id"])
    df = df[df["article_id"].isin(article_ids)]
    return df


def create_users_table(df: pd.DataFrame):
    """
    Creates the user table from the user history data.

    The password must be hashed, see: https://docs.djangoproject.com/en/4.0/topics/auth/passwords/
    To generate the hardcoded hash you see in this function, run
        python manage.py shell
        from django.contrib.auth.hashers import make_password
        make_password("userpassword", salt="abc")
    This will hash 'userpassword'.  All users will have this password.
    """
    user_df = pd.DataFrame(
        {"user_id": df['user_id'].drop_duplicates().sort_values(ascending=True)}
    ).reset_index(drop=True)
    # populate fields that would be required at signup
    user_df['username'] = user_df["user_id"].apply(lambda x: f"user{x}")
    user_df['email'] = user_df["user_id"].apply(lambda x: f"user{x}@user.com")
    user_df['password'] = 'pbkdf2_sha256$216000$abc$b1fH5TMQQ5QIkK0u/IqNHj47+q/l4ISTt/Hq3G3XGOE='
    # populate other fields
    user_df['is_staff'] = False
    user_df['is_active'] = True
    return user_df


def convert_df_to_json_fixture(df: pd.DataFrame, django_data_model: str, outfile_path: str, primary_key: str = None):
    """
    Converts a Pandas dataframe to a JSON fixture for Django.  Django fixtures come in the following format:
    [
        {
            "model": "app_name.model_name",
            "pk": 0,
            "fields": {
                "field_1": "value_1",
                ...
            }
        },
        ...
    ]
    """
    if "." not in django_data_model:
        raise ValueError(f"Expected a django_data_model like 'app_name.model_name', but received {django_data_model}")

    df = df.copy()  # all changes made in this function should be separate from the original
    df["model"] = django_data_model

    # if primary_key is not a column in the Pandas df, create one
    if primary_key is None or primary_key not in df.columns:
        p_keys = [i for i in range(len(df))]
        df["pk"] = p_keys
    else:
        df.rename(columns={primary_key: "pk"}, inplace=True)

    # some data types do not play well with JSON serialization, so convert them to strings
    cols_to_convert_to_str = df.select_dtypes(include=['datetime', 'datetime64', 'datetimetz']).columns
    for col in cols_to_convert_to_str:
        df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")  # Django expects date-times in this format
        df[col] = df[col].astype(str).str.lower().replace("nan", "2000-01-01 00:00:00")  # Set default date for missing timestamps

    # re-structure the data to fit the format required by Django
    df_records = []
    for row in df.iterrows():
        fields = [{col: val} for col, val in row[1].items() if col not in ['model', 'pk']]
        row_dict = {
            "model": row[1]['model'],
            "pk": row[1]['pk'],
            "fields": {col: val for d in fields for col, val in d.items()}
        }
        df_records.append(row_dict)

    with open(outfile_path, "w") as outfile:
        json.dump(df_records, outfile, indent=4, default=str)


if __name__ == "__main__":

    # create a place to hold the downloaded data
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    # download the MIND dataset to the data folder, if it doesn't exist already
    download_mind_data(data_path=data_folder + "/")

    # read the data to pandas for manipulation
    behavior_df = pd.read_csv(
        "data/train/behaviors.tsv",
        sep="\t",
        names=["impression_id", "user_id", "time", "history", "impressions"]
    )
    article_df = pd.read_csv(
        "data/train/news.tsv",
        sep="\t",
        names=["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        usecols=["article_id", "title", "abstract", "category", "url"]
    )

    # convert article and user IDs to integers
    behavior_df, _ = convert_id_to_int(df=behavior_df, id_col="user_id")
    article_df, article_id_to_new_aid_map = convert_id_to_int(df=article_df, id_col="article_id")

    # get timestamps for articles that were impressions
    imp_article_times = get_impression_article_times(behavior_df=behavior_df)
    imp_article_times['article_id'] = imp_article_times['article_id'].map(article_id_to_new_aid_map)
    imp_article_times = imp_article_times[imp_article_times['article_id'].notna()]
    imp_article_times['article_id'] = imp_article_times['article_id'].astype(int)
    imp_article_times = remove_missing_article_ids(df=imp_article_times, article_df=article_df)
    article_df['timestamp'] = article_df['article_id'].map(
        dict(zip(imp_article_times['article_id'], imp_article_times['time']))
    )

    # set the date for Django to use based on the one with the most articles
    # os.environ["TODAYS_DATE"] = article_df['timestamp'].dt.date.value_counts(
    #     sort=True, ascending=False
    # ).head(1).index[0].strftime("%Y-%m-%d")

    # convert articles_df to a JSON fixture for Django
    convert_df_to_json_fixture(
        df=article_df,
        django_data_model="articles.article",
        primary_key="article_id",
        outfile_path="articles/fixtures/articles.json",
    )

    # get the user-article history
    uap = get_user_article_pairs(behavior_df=behavior_df)
    uap['article_id'] = uap['article_id'].map(article_id_to_new_aid_map)
    uap = uap[uap['article_id'].notna()]
    uap['article_id'] = uap['article_id'].astype(int)
    uap = remove_missing_article_ids(df=uap, article_df=article_df)

    # derive the users table from the user-article history
    user_df = create_users_table(df=uap)

    # convert user_df to a JSON fixture for Django
    convert_df_to_json_fixture(
        df=user_df,
        django_data_model="users.customuser",
        outfile_path="users/fixtures/customuser.json",
        primary_key="user_id"
    )

    # convert user-article history to a JSON fixture for Django
    convert_df_to_json_fixture(
        df=uap,
        django_data_model="articles.userhistory",
        outfile_path="articles/fixtures/userhistory.json",
        primary_key=None
    )

    # get the top n stories for each day
    topn = int(os.getenv("NBR_TOP_STORIES", default=10))
    eligible_articles = article_df[~article_df['timestamp'].isnull()].copy()
    eligible_articles['timestamp'] = eligible_articles['timestamp'].dt.date
    eligible_articles = pd.merge(
        left=eligible_articles,
        right=uap[['article_id', 'id']],
        how='inner',
        on='article_id'
    )
    top_stories = eligible_articles.groupby(
        ['timestamp', 'article_id']
    )['id'].count().reset_index().rename(columns={"id": "clicks"}).sort_values(
        ['timestamp', 'clicks'], ascending=[True, False]
    )
    top_stories = top_stories.groupby(['timestamp'])['article_id', 'clicks'].apply(
        lambda g: g.nlargest(topn, 'clicks')
    ).reset_index().drop("level_1", axis=1)
    top_stories = pd.merge(
        left=top_stories.rename(columns={"timestamp": "date"}),
        right=article_df.drop("timestamp", axis=1),
        how='left',
        on='article_id'
    )

    # convert top_stories to a JSON fixture for Django
    convert_df_to_json_fixture(
        df=top_stories,
        django_data_model="articles.topstories",
        outfile_path="articles/fixtures/topstories.json",
        primary_key=None
    )
