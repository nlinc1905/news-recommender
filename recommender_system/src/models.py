import numpy as np
import pandas as pd
from pathlib import Path
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel

from .utils import download_mind_data, convert_id_to_int


class RecommendersNewsrecModel:

    def __init__(self, iterator, model_type: str, seed: int = 14, epochs: int = 1, batch_size: int = 32):
        # Validation checks
        if model_type not in ['lstur', 'naml', 'npa', 'nrms']:
            raise ValueError(f"model_type should be 1 of 'lstur', 'naml', 'npa', 'nrms', but received {model_type}")

        # Set attributes
        self.iterator = iterator
        self.model_type = model_type
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        # Placeholders
        self.file_paths = None
        self.model = None

    def _prepare_training_and_eval_data(self, **kwargs):
        """Downloads and cleans training & validation data."""
        self.file_paths = download_mind_data(model_type=self.model_type, **kwargs)

    def prepare_test_data(self):
        """
        The training set is given new article IDs before being saved in Django's DB.
        This function resets the training data's article IDs to match what will be received
        in requests from Django, and saves the result to the /test folder.
        """
        article_dft = pd.read_csv(
            "data/train/news.tsv",
            sep="\t",
            names=["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities",
                   "abstract_entities"]
        )

        # remove tabs from text
        article_dft['title'] = article_dft['title'].str.replace("\t", "")
        article_dft['abstract'] = article_dft['abstract'].str.replace("\t", "")

        # convert article and user IDs to integers
        article_dft, tid_to_new_aid_map = convert_id_to_int(df=article_dft, id_col="article_id")

        # save to test, as the article IDs there will match the test data
        article_dft.to_csv("data/test/news.tsv", sep="\t", index=False, header=False)

    def _prepare_model(self):
        """Sets up the model's hyperparameters."""
        file_paths = {k: str(v) for k, v in self.file_paths.items()}
        hparams = prepare_hparams(
            yaml_file=file_paths['yaml'],
            wordEmb_file=file_paths['glove_embed'],
            wordDict_file=file_paths['word_dict'],
            userDict_file=file_paths['user_dict'],
            batch_size=self.batch_size,
            epochs=self.epochs,
            show_step=10
        )
        if self.model_type == "nrms":
            self.model = NRMSModel(hparams=hparams, iterator_creator=self.iterator, seed=self.seed)

    def train(self):
        """Downloads training data, sets hyperparameters, and fits the model to the training data."""
        self._prepare_training_and_eval_data()
        self._prepare_model()
        self.model.fit(
            self.file_paths['train_news'],
            self.file_paths['train_behaviors'],
            self.file_paths['valid_news'],
            self.file_paths['valid_behaviors'],
        )

    def eval(self):
        """Evaluates the model on validation dataset"""
        if self.model is None:
            raise TypeError(f"No model found.  Has it been trained yet?")
        return self.model.run_eval(self.file_paths['valid_news'], self.file_paths['valid_behaviors'])

    def save(self, models_path: Path):
        """Saves the model weights to the provided path."""
        self.model.model.save_weights(models_path / f"{self.model_type}/ckpt")

    def load(self, models_path: Path):
        """Loads the model weights from the provided path."""
        self._prepare_training_and_eval_data()
        self._prepare_model()
        self.model.model.load_weights(models_path / f"{self.model_type}/ckpt")

    @staticmethod
    def parse_incoming_request(data):
        """
        Parses incoming JSON request into a usable format for the model.

        The newsrec models use a MINDIterator that can only accept input from files, so this function translates
        incoming data into a dataframe and dumps it to a tsv file so that the iterator can read it and inference
        can be performed.
        """
        behavior_df = pd.DataFrame(data['behavior'])
        # flatten the df to one line by collapsing article_id into history
        behavior_df['history'] = " ".join(behavior_df['article_id'].astype(str).tolist())
        behavior_df = behavior_df.drop('article_id', axis=1).drop_duplicates()
        # add the article IDs as impressions
        behavior_df['impressions'] = " ".join([str(a['article_id']) + "-0" for a in data['articles']])
        # the model does not need these fields, but the model iterator still expects them
        behavior_df['impression_id'] = "1"
        behavior_df['time'] = "1"
        # reorder columns to match what iterator expects
        behavior_df = behavior_df[['impression_id', 'user_id', 'time', 'history', 'impressions']]
        behavior_df.to_csv("data/test/behaviors.tsv", sep="\t", index=False, header=False)

    def test_predict(self):
        behavior_df = pd.read_csv(
            "data/valid/behaviors.tsv",
            sep="\t",
            names=["impression_id", "user_id", "time", "history", "impressions"]
        )
        article_df = pd.read_csv(
            "data/valid/news.tsv",
            sep="\t",
            names=["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities",
                   "abstract_entities"],
            # usecols=["article_id", "title", "abstract"]
        )
        behavior_df['impression_id'] = "1"
        behavior_df['time'] = "1"
        # behavior_df['impressions'] = behavior_df['impressions'].str.replace("-0", "").replace("-1", "")
        article_df['category'] = "1"
        article_df['subcategory'] = "1"
        article_df['url'] = "1"
        article_df['title_entities'] = '[]'
        article_df['abstract_entities'] = '[]'
        article_df['title'] = article_df['title'].str.replace("\t", "")
        article_df['abstract'] = article_df['abstract'].str.replace("\t", "")
        behavior_df.to_csv("data/test/behaviors.tsv", sep="\t", index=False, header=False)
        article_df.to_csv("data/test/news.tsv", sep="\t", index=False, header=False)
        '''
        with tf.io.gfile.GFile("data/test/news.tsv", "r") as rd:
            for li, line in enumerate(rd):
                ll = len(line.strip("\n").split('\t'))
                if ll != 8:
                    print(li, line)
                    breakpoint()
        '''
        group_impr_indices, group_label, group_preds = self.model.run_fast_eval(
            "data/test/news.tsv",
            "data/test/behaviors.tsv",
        )
        pred_rank = [(np.argsort(np.argsort(pred)[::-1]) + 1).tolist() for pred in group_preds]
        breakpoint()

    def predict_proba(self, data):
        """Predicts the score of articles"""
        self.parse_incoming_request(data=data)
        group_impr_indices, group_label, group_preds = self.model.run_fast_eval(
            "data/test/news.tsv",
            "data/test/behaviors.tsv",
        )
        return dict(zip([a['article_id'] for a in data['articles']], group_preds[0]))

    def predict(self, data):
        """Predicts the rank of articles"""
        self.parse_incoming_request(data=data)
        group_impr_indices, group_label, group_preds = self.model.run_fast_eval(
            "data/test/news.tsv",
            "data/test/behaviors.tsv",
        )
        pred_rank = [(np.argsort(np.argsort(pred)[::-1]) + 1).tolist() for pred in group_preds]
        return dict(zip([a['article_id'] for a in data['articles']], pred_rank[0]))
