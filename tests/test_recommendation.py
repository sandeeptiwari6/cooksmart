import unittest
from recommendation import RecipeRecommender as rr
from helpers import DataFormatError, QueryError
from unittest.mock import Mock
import os
import pandas as pd
import numpy as np

class test_rr_initialization(unittest.TestCase):

    #invalid file path
    def test_filepath(self):
        self.assertRaises(FileNotFoundError,rr,"invalid.csv")

    #invalid column names
    def test_invalid_format(self):
        data = {'recipe_name':['Chicken dish', 'pasta'], 
        'ingredients':["chiecken", "pasta"]}
        data =pd.DataFrame(data)
        data.to_csv("invalid_data_format.csv") 
        self.assertRaises(DataFormatError,rr,"invalid_data_format.csv")
        os.remove('invalid_data_format.csv')


    # title tfidf is correct size?
    def test_recipe_ingredient_matrix(self):
        r = rr()
        self.assertEqual(len(r.data),r.recipe_ingredient_matrix.shape[0])

        r =rr('../data/cleaned-data_recipe.csv')
        self.assertEqual(len(r.data),r.recipe_ingredient_matrix.shape[0])
    
    def test_title_tfidf(self):
        r = rr()
        self.assertEqual(len(r.data),r.title_tfidf.shape[0])

        r =rr('../data/cleaned-data_recipe.csv')
        self.assertEqual(len(r.data),r.title_tfidf.shape[0])


class test_rr_fit(unittest.TestCase):
    #stay with defaults - fit shouldn't be called
    def test_defaults(self):
        pass

    #provide filepath - fit should be called once

    #change default num topics - fit should be called once

    #recipe topic distribution
    def test_recipe_topic_distribution(self):
        r =rr()
        r.fit()
        self.assertEqual(r.n_components,r.recipe_topic_dist.shape[1])
        self.assertEqual(len(r.data),r.recipe_topic_dist.shape[0])
    #title topic distribution
    def test_title_topic_distribution(self):
        r =rr()
        r.fit()
        self.assertEqual(r.n_components,r.title_topic_dist.shape[1])
        self.assertEqual(len(r.data),r.title_topic_dist.shape[0])

class test_rr_get_recommendations(unittest.TestCase):

    def test_input_type(self):
        r =rr()
        r.fit()

        with self.assertRaises(QueryError):
            rr.get_recommendations(self,["1",3])


    def test_input_count(self):
        r =rr()
        r.fit()

        with self.assertRaises(QueryError):
            rr.get_recommendations(self,["1","2"])

    # check size of ingredient topic dist

    #length of sorted index

    # maek sure # of recommendations =n

    #make sure that scores are descending
    # def test_sorted_order(self):
    #     r=rr()
    #     r.fit()



class test_rr_similarity(unittest.TestCase):

    def test_score_array(self):
        r=rr()
        r.recipe_topic_dist = np.matrix([[0.3,0.2,0.2,0.1,0.2],
                                         [0.2,0.1,0.4,0.3,0],
                                         [0,0.5,0.1,0.4,0]])
        r.title_topic_dist = np.matrix([[0.4,0.1,0.1,0.2,0.2],
                                         [0.15,0.15,0.4,0.3,0],
                                         [0,0.4,0.2,0.4,0]])

        r.input_ingredients_topic_dist = np.matrix([[0.3,0.2,0.2,0.1,0.2]])

        score = r.recipe_similarity()
        self.assertEqual(len(score), 3)
                            


class test_rr_visualize_recommendation(unittest.TestCase):
    def test_visualize_call(self):
        r=rr()
        with self.assertRaises(AttributeError):
            r.visualize_recommendation()

class test_rr_pickle(unittest.TestCase):
    def test_pickle(self):
        r=rr()



if __name__ == '__main__':
    unittest.main()