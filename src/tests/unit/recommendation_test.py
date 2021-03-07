import unittest
from recommendation import RecipeRecommender as rr, DataFormatError, QueryError
from unittest.mock import Mock

class test_rr_initialization(unittest.TestCase):

    #invalid file path
    def test_filepath(self):
        self.assertRaises(FileNotFoundError,rr,"invalid.csv")

    #invalid file, pandas can't open?

    #invalid column names
    def test_invalid_format(self):
        self.assertRaises(DataFormatError,rr,"invalid_data_format.csv")

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

class test_rr_visualize_recommendation(unittest.TestCase):
    def test_visualize_call(self):
        r=rr()
        with self.assertRaises(AttributeError):
            r.visualize_recommendation()



if __name__ == '__main__':
    unittest.main()