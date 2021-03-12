# urls to pickled objects on github
base1 = "https://raw.githubusercontent.com/sandeeptiwari6/cooksmart/"
base2 = "https://github.com/sandeeptiwari6/cooksmart/blob/main/"
urls = {
    "DATA_URL": base1+"main/data/cleaned-data_recipe.csv",
    "LDA_URL": base2+"cooksmart/pickles/lda.pickle?raw=true",
    "TFIDF_URL": base2+"cooksmart/pickles/vectorizer.pickle?raw=true",
    "REC_TOP_URL": base2+"cooksmart/pickles/recipe_topics.pickle?raw=true",
    "TIT_TOP_URL": base2+"cooksmart/pickles/title_topics.pickle?raw=true"
}
