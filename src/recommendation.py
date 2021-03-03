import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class RecipeRecommender:
    def __init__(self, filepath='../data/cleaned-data_recipe.csv', max_df=0.6, min_df=2):
        """
        Creates an instance of Recipe Recommender
        Reads in formatted csv file and vectorize recipes

        Input: 
        filepath (str)
        max_df (float): max document freq accepted by tfidf vectorizer
        min_df (float): min document freq accepted by tfidf vectorizer

        Output:
        None
        """
        self.filepath =filepath
        self.tfidf_vect = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
        
        #TODO: make sure valid path
        #TODO:  make sure dataframe  is correctly formatted
        self.data = pd.read_csv(filepath)

        #TODO: additional cleaning - bigrams, trigrams, lemmatization, lower case
        self.recipe_ingredient_matrix = self.tfidf_vect.fit_transform(self.data['ingredients'].values.astype('U'))
        self.title_tfidf = self.tfidf_vect.transform(self.data['recipe_name'])
    
    def fit(self,n_components=8):
        """
        Fits an LDA model to the dataset provided
        For each recipe, creates a topic distribution for the ingredients as well as for the recipe name
        
        Input: n_components (INT): number of topics desired
        
        Output: None 
        """
        self.LDA = LatentDirichletAllocation(n_components=n_components, random_state=42)
        self.LDA.fit_transform(self.recipe_ingredient_matrix)
        self.recipe_topic_dist = np.matrix(self.LDA.transform(self.recipe_ingredient_matrix))
        self.title_topic_dist = self.LDA.transform(self.title_tfidf)

    def recipe_similarity(self,w_title=0.2,w_text=0.3):
        """
        Compares topic distribution of input ingredients against topic distribution of 
        each recipe in dataset, calculates "similarity"

        Input: 
        w_title (float between 0 and 1)
        w_text (float between 0 and 1)

        Output: 
        scores (np.array)
        """
        scores = self.recipe_topic_dist*self.input_ingredients_topic_dist.T *w_text
        scores += self.title_topic_dist * self.input_ingredients_topic_dist.T * w_title

        #TODO: try other similarities, normalize scores?
        scores = np.squeeze(np.asarray(scores))
        return scores

    def get_recommendations(self,input_ingredients,n=3):
        """
        Takes in a string of space separated ingredients, and returns 
        a dataframe of the recipes most "similar"

        Input: 
        input_ingredients (list of strings)
        n (int): number of recipes to return
        """
        self.input_ingredients = input_ingredients
        self.input_ingredients_tfidf = self.tfidf_vect.transform([' '.join(self.input_ingredients)])
        self.input_ingredients_topic_dist = np.matrix(self.LDA.transform(self.input_ingredients_tfidf))

        scores = self.recipe_similarity()
        sorted_index = np.argsort(scores)[::-1]
        return (self.data.iloc[sorted_index, :])

    def visualize_fit(self):
        pass

    def visualize_recommendation(self):
        pass

    def pickle_model(self):
        pass

