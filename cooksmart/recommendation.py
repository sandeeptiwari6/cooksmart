import pandas as pd
import numpy as np
import pickle
import os
import webbrowser
import time

import pyLDAvis
import pyLDAvis.sklearn
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from .helpers import DataFormatError, QueryError, is_valid_recipe_df


# urls to pickled objects on github
urls = {
    "DATA_URL": "https://raw.githubusercontent.com/sandeeptiwari6/recommender-system-515/main/data/cleaned-data_recipe.csv",
    "LDA_URL": "https://github.com/sandeeptiwari6/recommender-system-515/blob/main/src/pickles/lda.pickle?raw=true",
    "TFIDF_URL": "https://github.com/sandeeptiwari6/recommender-system-515/blob/main/src/pickles/vectorizer.pickle?raw=true",
    "REC_TOP_URL": "https://github.com/sandeeptiwari6/recommender-system-515/blob/main/src/pickles/recipe_topics.pickle?raw=true",
    "TIT_TOP_URL": "https://github.com/sandeeptiwari6/recommender-system-515/blob/main/src/pickles/title_topics.pickle?raw=true"
}


class RecipeRecommender:
    def __init__(self, filepath=None, max_df=0.6, min_df=2):
        """
        Creates an instance of Recipe Recommender
        Initialises : filepath, tfidf_vect, data, recipe_ingredient_matrix, 
        title_tfidf
        Reads in formatted csv file and vectorize recipes
        Input:
        filepath (str)
        max_df (float): max document freq accepted by tfidf vectorizer
        min_df (float): min document freq accepted by tfidf vectorizer
        Output:
        None
        """

        self.filepath = filepath
        self.tfidf_vect = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                          stop_words='english')
        if self.filepath is None:
            self.data = pd.read_csv('../data/cleaned-data_recipe.csv')
            with open('pickles/recipe_topics.pickle', 'rb') as f:
                self.recipe_ingredient_matrix = pickle.load(f)
            with open('pickles/vectorizer.pickle', 'rb') as f:
                self.tfidf_vect = pickle.load(f)
            self.title_tfidf = self.tfidf_vect.transform(
                self.data['recipe_name'])

        else:
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"{filepath} is not a valid path to a dataset")
            self.data = pd.read_csv(filepath)
       
            if not is_valid_recipe_df(self.data):
                raise DataFormatError("Inputted csv is incorrectly formatted")

            self.recipe_ingredient_matrix = self.tfidf_vect.fit_transform(
                self.data['ingredients'].values.astype('U'))
            self.title_tfidf = self.tfidf_vect.transform(
                                            self.data['recipe_name'])

    def fit(self, n_components=10):
        """
        Fits an LDA model to the dataset provided
        For each recipe, creates a topic distribution for the ingredients as
        well as for the recipe name

        Input: n_components (INT): number of topics desired

        Output: None
        """
        if self.filepath is None and n_components == 10:
            with open('pickles/lda.pickle', 'rb') as f:
                self.LDA = pickle.load(f)
            with open('pickles/title_topics.pickle', 'rb') as f:
                self.title_topic_dist = pickle.load(f)
            # save as pickle
            self.recipe_topic_dist = np.array(self.LDA.transform(
                                                self.recipe_ingredient_matrix))
        else:
            self.LDA = LatentDirichletAllocation(n_components=n_components,
                                                 random_state=42)
            self.LDA.fit_transform(self.recipe_ingredient_matrix)
            self.recipe_topic_dist = np.array(self.LDA.transform(
                                                self.recipe_ingredient_matrix))
            self.title_topic_dist = self.LDA.transform(self.title_tfidf)

    def recipe_similarity(self, w_title=0.2, w_text=0.3):
        """
        Compares topic distribution of input ingredients against topic
        distribution of each recipe in dataset, calculates "similarity"

        Input:
        w_title (float between 0 and 1)
        w_text (float between 0 and 1)
        Output:
        scores (np.array)
        """
        scores = self.recipe_topic_dist @ self.input_ingredients_topic_dist.T * w_text
        scores += self.title_topic_dist @ self.input_ingredients_topic_dist.T * w_title

        # TODO: try other similarities, normalize scores?
        scores = np.squeeze(np.asarray(scores))
        return scores

    def get_recommendations(self, input_ingredients: list, n=3):
        """
        Takes in a string of space separated ingredients, and returns
        a dataframe of the recipes most "similar"

        Input:
        input_ingredients (list of strings)
        n (int): number of recipes to return
        """

        for ingredient in input_ingredients:
            if not isinstance(ingredient, str):
                raise QueryError("Ingredients must be strings")
        if len(input_ingredients) < 5:
            raise QueryError("Input atleast 5 Ingredients")

        input_ingredients_tfidf = self.tfidf_vect.transform(
                                                [' '.join(input_ingredients)])
        self.input_ingredients_topic_dist = np.array(self.LDA.transform(
                                                input_ingredients_tfidf))
        scores = self.recipe_similarity()
        sorted_index = np.argsort(scores)[::-1]
        return self.data.iloc[sorted_index, :].head(n)

    def visualize_fit(self):
        """
        Opens browser with visualization of fitted LDA model.
        :return: None
        """
        topic_viz = pyLDAvis.sklearn.prepare(self.LDA,
                                             self.recipe_ingredient_matrix,
                                             self.tfidf_vect)
        save_file = 'lda-results.html'
        pyLDAvis.save_html(topic_viz, save_file)
        webbrowser.open('file://' + os.path.realpath(save_file))
        time.sleep(1)
        os.remove(save_file)

    def visualize_recommendation(self):
        """
        Opens browser with visualization of topic probability distribution of
        latest query.
        :return: None
        """
        try:
            topic_dist = np.squeeze(np.asarray(
                self.input_ingredients_topic_dist))
            topics = [f"Topic {i}" for i in range(1, len(topic_dist)+1)]
            df = pd.DataFrame({'topics': topics, 'probability': topic_dist})
            fig = px.line_polar(df, r="probability", theta="topics",
                                line_close=True, 
                                color_discrete_sequence=px.colors.sequential.Plasma_r,
                                template="plotly_dark")
            fig.show()
        except AttributeError:
            raise AttributeError(" Call `get_recommendations` first")

    def pickle_model(self, lda_file="pickles/lda.pickle"):
        """
        Pickles current LDA model to filepath defined by user.
        :param lda_file: filepath as string
        :return: None
        """
        with open(lda_file, "wb") as f:
            print(f"Saving LDA model to {lda_file}...")
            pickle.dump(self.LDA, f)


# if __name__ == "__main__":
#     rr = RecipeRecommender()
#     rr.fit()
#     query = ["pepper", "chicken", "salt", "vinegar", "tomato", "cheese"]

#     rr.get_recommendations(query)
#     # rr.visualize_fit()
#     # rr.visualize_recommendation()
