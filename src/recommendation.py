import pandas as pd
import numpy as np
from functools import reduce

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class RecipeRecommender:
    def __init__(self, input_ingredients, filename='../data/cleaned-data_recipe.csv', max_df=0.6, min_df=2):
        """

        :param input_ingredients: list of ingredients
        :param filename: name of dataset used for recommendation
        :param max_df:
        :param min_df:
        """
        # assert len(input_ingredients) >= 5
        # input ingredients must have at least 5 ingredients

        self.input_ingredients = input_ingredients
        self.filename = filename

        # TODO: create bigrams
        self.tfidf_vect = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english')

        # TODO: test that dataframe is correctly formatted


    def fit(self, n_components=5):
        self.data = pd.read_csv(self.filename)
        self.doc_term_matrix = self.tfidf_vect.fit_transform(self.data['ingredients'].values.astype('U'))

        self.LDA = LatentDirichletAllocation(n_components=n_components, random_state=42)
        self.LDA.fit(self.doc_term_matrix)
        self.doc_topic_dist_unnormalized = np.matrix(self.LDA.transform(self.tfidf_vect.transform(self.data['ingredients'])))

    def docs_by_tops(self, top_mat, topic_range=(0, 0), doc_range=(0, 2)):
        for i in range(topic_range[0], topic_range[1]):
            topic_scores = pd.Series(top_mat[:, i])
            doc_index = topic_scores.sort_values(ascending=False)[doc_range[0]:doc_range[1]].index
            for j, index in enumerate(doc_index, doc_range[0]):
                print('Topic #{}'.format(i),
                      '\nDocument #{}'.format(j),
                      '\nTopic Score: {}\n\n'.format(topic_scores[index]),
                      self.data['ingredients'][index], '\n\n')

    # def recommend(self):
        # X_test = self.tfidf_vect.transform(self.input_ingredients)
        # doc_topic_dist_unnormalized = np.matrix(self.LDA.transform(X_test))
        #
        # doc_topic_dist = doc_topic_dist_unnormalized / doc_topic_dist_unnormalized.sum(axis=1)

        # self.docs_by_tops(self.lda_fit, (0, 3), (0, 3))


    # query functions

    def qweight_array(self, query_length, qw_array=[1]):
        '''Returns descending weights for ranked query ingredients'''
        if query_length > 1:
            to_split = qw_array.pop()
            split = to_split / 2
            qw_array.extend([split, split])
            return self.qweight_array(query_length - 1, qw_array)
        else:
            return np.array(qw_array)

    def ranked_query(self, query):
        '''Called if query ingredients are ranked in order of importance.
        Weights and adds each ranked query ingredient vector.'''
        query = [[q] for q in query]  # place words in seperate documents
        q_vecs = [self.tfidf_vect.transform(q) for q in query]
        qw_array = self.qweight_array(len(query), [1])
        q_weighted_vecs = q_vecs * qw_array
        q_final_vector = reduce(np.add, q_weighted_vecs)
        return q_final_vector

    def overall_scores(self, query_vector):
        '''Calculates Query Similarity Scores against recipe title, instructions, and keywords.
        Then returns weighted averages of similarities for each recipe.'''
        title_tfidf = self.tfidf_vect.transform(self.data['recipe_name'])
        title_vect = self.LDA.transform(title_tfidf)
        # text_tfidf = self.tfidf_vect.transform(self.data['ingredients'])
        w_title = .2
        w_text = .3
        final_scores = self.doc_topic_dist_unnormalized * query_vector.T * w_text
        final_scores += title_vect * query_vector.T * w_title
        # final_scores += tags_tfidf * query_vector.T * w_categories
        return final_scores

    def print_recipes(self, index, query, recipe_range):
        '''Prints recipes according to query similary ranks'''
        print('Search Query: {}\n'.format(query))
        for i, index in enumerate(index, recipe_range[0]):
            print('Recipe Rank: {}\t'.format(i + 1), self.data.loc[index, 'recipe_name'], '\n')
            print('Ingredients:\n{}\n '.format(self.data.loc[index, 'ingredients']))
            # print('Instructions:\n{}\n'.format(self.data.loc[index, 'instructions']))

    def Search_Recipes(self, query_ranked=False, recipe_range=(0, 3)):
        '''Master Recipe Search Function'''
        query = self.input_ingredients[0].split()
        if query_ranked == True:
            q_vector = self.ranked_query(query)
        else:
            q_vector = self.tfidf_vect.transform([' '.join(query)])
        q_vector = np.matrix(self.LDA.transform(q_vector))
        recipe_scores = self.overall_scores(q_vector)
        recipe_scores = np.squeeze(np.asarray(recipe_scores))
        sorted_index = np.argsort(recipe_scores)[::-1]
        print(self.data.iloc[sorted_index, :])
        # pd.Series(recipe_scores.T[0]).sort_values(ascending=False)[recipe_range[0]:recipe_range[1]].index
        # return self.print_recipes(sorted_index, query, recipe_range)


if __name__ == "__main__":
    import time
    start = time.time()
    print("initializing recommender...")
    rr = RecipeRecommender(input_ingredients=["vanilla flour eggs cheese pepper"])
    print(time.time() - start)
    print('fitting data...')
    rr.fit()
    print(time.time() - start)
    print('making recommendations...')
    rr.Search_Recipes()
    print(time.time() - start)
