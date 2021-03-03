import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class RecipeRecommender:
    def __init__(self, filename='../data/cleaned-data_recipe.csv', max_df=0.6, min_df=2):
        self.filename =filename
        self.tfidf_vect = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
        # make sure valid path, data is correctly formatted
        self.data = pd.read_csv(filename)
        self.recipe_ingredient_matrix = self.tfidf_vect.fit_transform(self.data['ingredients'].values.astype('U'))
        self.title_tfidf = self.tfidf_vect.transform(self.data['recipe_name'])
        
    
    def fit(self,n_components=5):
        self.LDA = LatentDirichletAllocation(n_components=n_components, random_state=42)
        self.LDA.fit_transform(self.recipe_ingredient_matrix)
        self.recipe_topic_dist = np.matrix(self.LDA.transform(self.recipe_ingredient_matrix))
        self.title_topic_dist = self.LDA.transform(self.title_tfidf)

    def recipe_similarity(self,w_title=0.4,w_text=0.6):
        scores = self.recipe_topic_dist*self.input_ingredients_topic_dist.T *w_text
        scores += self.title_topic_dist * self.input_ingredients_topic_dist.T * w_title
        scores = np.squeeze(np.asarray(scores))
        return scores

    def get_recommendations(self,input_ingredients,n=3):
        self.input_ingredients = input_ingredients
        self.input_ingredients_tfidf = self.tfidf_vect.transform([' '.join(self.input_ingredients)])
        self.input_ingredients_topic_dist = np.matrix(self.LDA.transform(self.input_ingredients_tfidf))

        scores = self.recipe_similarity()
        sorted_index = np.argsort(scores)[::-1]
        return (self.data.iloc[sorted_index, :])

