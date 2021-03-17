import unittest
from cooksmart.recommendation import RecipeRecommender as rr
from cooksmart.exception_errors import DataFormatError, QueryError
from unittest.mock import Mock, patch
import os
import pandas as pd
import numpy as np


class test_rr_initialization(unittest.TestCase):
    # invalid file path
    def test_filepath(self):
        self.assertRaises(FileNotFoundError, rr, "invalid.csv")

    # invalid column names
    def test_invalid_format(self):
        data = {'recipe_name': ['Chicken dish', 'pasta'],
                'ingredients': ["chicken", "pasta"]}
        data = pd.DataFrame(data)
        data.to_csv("invalid_data_format.csv")
        self.assertRaises(DataFormatError, rr, "invalid_data_format.csv")
        os.remove('invalid_data_format.csv')

    # title tfidf is correct size?
    def test_recipe_ingredient_matrix(self):
        r = rr()
        self.assertEqual(len(r.data), r.recipe_ingredient_matrix.shape[0])

        r = rr("cooksmart/tests/test.csv")
        self.assertEqual(len(r.data), r.recipe_ingredient_matrix.shape[0])

    def test_title_tfidf(self):
        r = rr()
        self.assertEqual(len(r.data), r.title_tfidf.shape[0])

        r = rr("cooksmart/tests/test.csv")
        self.assertEqual(len(r.data), r.title_tfidf.shape[0])


class test_rr_fit(unittest.TestCase):
    # provide filepath - LDA should be called once
    @patch('cooksmart.recommendation.LatentDirichletAllocation')
    def test_LDA_calls1(self, mock_A):
        r = rr("cooksmart/tests/test.csv")
        r.fit()
        self.assertEqual(mock_A.call_count, 1)

    # provide no filepath - LDA should be called zero times
    @patch('cooksmart.recommendation.LatentDirichletAllocation')
    def test_LDA_calls2(self, mock_A):
        r = rr()
        r.fit()
        self.assertEqual(mock_A.call_count, 0)

    # change default num topics - LDA should be called once
    @patch('cooksmart.recommendation.LatentDirichletAllocation')
    def test_LDA_calls3(self, mock_A):
        r = rr()
        r.fit(n_components=5)
        self.assertEqual(mock_A.call_count, 1)

    # recipe topic distribution
    def test_recipe_topic_distribution(self):
        r = rr()
        r.fit()
        self.assertEqual(10, r.recipe_topic_dist.shape[1])
        self.assertEqual(len(r.data), r.recipe_topic_dist.shape[0])

    # title topic distribution
    def test_title_topic_distribution(self):
        r = rr()
        r.fit()
        self.assertEqual(10, r.title_topic_dist.shape[1])
        self.assertEqual(len(r.data), r.title_topic_dist.shape[0])


class test_rr_get_recommendations(unittest.TestCase):

    def test_input_type(self):
        r = rr()
        r.fit()

        with self.assertRaises(QueryError):
            rr.get_recommendations(self, ["1", 3])

    def test_input_count(self):
        r = rr()
        r.fit()

        with self.assertRaises(QueryError):
            rr.get_recommendations(self, ["1", "2"])

    @patch("cooksmart.recommendation.RecipeRecommender.recipe_similarity")
    def test_similarity_call(self, mock_similarity):
        r = rr()
        r.fit()
        r.get_recommendations(["pepper", "chicken", "salt", "vinegar",
                              "tomato", "cheese"])
        self.assertEqual(mock_similarity.call_count, 1)

    def test_output(self):
        r = rr()
        r.fit()
        scores = r.get_recommendations(["pepper", "chicken", "salt", "vinegar",
                                        "tomato", "cheese"], 8)
        self.assertEqual(len(scores), 8)


class test_rr_similarity(unittest.TestCase):

    def test_score_array(self):
        r = rr()
        r.recipe_topic_dist = np.array([[0.3, 0.2, 0.2, 0.1, 0.2],
                                        [0.2, 0.1, 0.4, 0.3, 0],
                                        [0, 0.5, 0.1, 0.4, 0]])
        r.title_topic_dist = np.array([[0.4, 0.1, 0.1, 0.2, 0.2],
                                       [0.15, 0.15, 0.4, 0.3, 0],
                                       [0, 0.4, 0.2, 0.4, 0]])

        r.input_ingredients_topic_dist = np.array([[0.3, 0.2, 0.2, 0.1, 0.2]])

        score = r.recipe_similarity()
        self.assertEqual(len(score), 3)


class test_rr_visualize_fit(unittest.TestCase):
    # pyLDAvis should be called once
    @patch('cooksmart.recommendation.pyLDAvis.sklearn.prepare')
    @patch('cooksmart.recommendation.pyLDAvis.save_html')
    # @patch('cooksmart.recommendation.os.remove')
    # @patch('cooksmart.recommendation.webbrowser')
    def test_pyLDAvis_calls1(self, mock_save, mock_prepare):
        r = rr()
        r.fit()
        r.visualize_fit()
        self.assertEqual(mock_prepare.call_count, 1)
        self.assertEqual(mock_save.call_count, 1)
        # self.assertEqual(mock_os.call_count, 1)


class test_rr_visualize_recommendation(unittest.TestCase):
    def test_visualize_call(self):
        r = rr()
        with self.assertRaises(AttributeError):
            r.visualize_recommendation()

    @patch('cooksmart.recommendation.px.line_polar')
    def test_fig_show(self, mock_show):
        r = rr()
        r.fit()
        r.get_recommendations(["pepper", "chicken", "salt", "vinegar",
                               "tomato", "cheese"])
        mock_show.return_value = Mock()
        r.visualize_recommendation()
        self.assertEqual(mock_show.call_count, 1)


class test_rr_pickle(unittest.TestCase):
    @patch('cooksmart.recommendation.pickle.dump')
    def test_pickle(self, mock_dump):
        r = rr()
        r.LDA = Mock()
        r.pickle_model("test.pkl")
        self.assertEqual(mock_dump.call_count, 1)


if __name__ == '__main__':
    unittest.main()
