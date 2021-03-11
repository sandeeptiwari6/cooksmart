## Data 515: Software Design for Data Science

## CookSmart - Recipe Recommendation system

## Introduction 

The goal of this project is to use the recipe dataset available in kaggle to build a recipe recommendation system that allows users to search for recipes based on the ingredients list. The recipe dataset is cleaned and parsed using a util file. Using the cleaned dataset, the recommendation system will perform unsupervised topic modeling on the recipes by vectorizing the recipe name, ingredients and user-input ingredients list using TF-IDF. The vectorized representations are then provided as an input to the LDA model to generate a topic probability distribution. Finally, a similarity score is generated for each recipe in the dataset based on the user-input ingredients list and the recipes are recommended based on highest scores. 


## Objective

- Given any recipe dataset, build a model to find the most relevant recipes that match the user’s ingredients list.
- Perform unsupervised Topic Modeling on the recipes to group recipes into topics.
- Rank the ingredients based on its frequency i.e. each successive ingredient in the list is weighted incrementally less.
- Create a search algorithm that utilizes similarity scoring to rank recipes according to the greatest similarity to the user-input list of ingredients and returns recipe recommendations based on the scores. 


## Dataset Description:

The kaggle dataset used for our recommendation system was created by scraping AllRecipes.com, a popular social network recipe site. For each recipe, the dataset contains a corresponding row with ecipe id, name, average ratings of reviewers, image url, review nums, ingredients, cooking directions, nutritions, and reviews. . We have pre-filtered the dataset to exclude records which contains repeated ingredients, no images or zero reviews to ensure data quality. The final dataset includes 49,698 recipes with 38,131 ingredients representing a varied range of cuisines and tastes. Given below is the description of the columns available in the dataset.

| Column | Datatype | Value |
| ------ | -------- | ----- |
| recipe_id | int | A unique id associated with each recipe |
| recipe_name | string | Name of the recipe |
| aver_rate | float | User rating for the recipe |
| image_url | string | Image url corresponding to the image associated with the recipe |
| review_nums | int | No. of reviews for that recipe |
| ingredients | string | A list of ingredients associated with the recipe |
| cooking_directions | string | Instructions for making the recipe |
| nutritions | string | Nutritions associated with the recipe |
| reviews | string | Reviews associated with the recipe |


## Dataset Requirements that need to be met for new datasets

Incase you decide to use your own dataset other than the one listed above, please make sure your dataset meets the following requirements:

| Column | Datatype | Required |
| ------ | -------- | -------- |
| recipe_name | string | Yes |
| ingredients | list of strings seperated by comma | Yes |
| cooking_directions | string | Yes |

If any of the above columns are not available or if there is a column name or data type mismatch the recommender function will fail.

## Features

### Data Features:

- Recipe Name
- Ingredients
- Cooking Directions

### Model Features:

- Search based on ingredients list (e.g.: [ingredient 1, ingredient 2, ingredient 3])
#### - Option to rank ingredients in order of ingredients. i.e. each successive ingredient in list is weighted incrementally less in the search query??
#### - what else??


## Project folder structure


## Installations Required

To install the recommender system you will need to install the package and run the below command:

python -m pip install --index-url https://test.pypi.org/simple/ Cooksmart-recommender-system

Once the download is complete, you should be able to import and use the Cooksmart-recommender-system on your machine.

## User guide on How to use the Cooksmart-recommender-system

- Once the installation is complete, run the below command in python:

import Cooksmart-recommender-system

This will import all of the functions included in the package which can then be called using the below command:

Cooksmart-recommender-system.RecipeRecommender()







