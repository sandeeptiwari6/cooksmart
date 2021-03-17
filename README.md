## Data 515: Software Design for Data Science
[![Build Status](https://api.travis-ci.com/sandeeptiwari6/cooksmart.svg?branch=main)](https://travis-ci.com/github/sandeeptiwari6/cooksmart)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## CookSmart - Recipe Recommendation system

## Introduction 

The goal of this project is to use the recipe dataset available in kaggle to build a recipe recommendation system that allows users to search for recipes based on the ingredients list. The recipe dataset is cleaned and parsed using a util file. Using the cleaned dataset, the recommendation system will perform unsupervised topic modeling on the recipes by vectorizing the recipe name, ingredients and user-input ingredients list using TF-IDF. The vectorized representations are then provided as an input to the LDA model to generate a topic probability distribution. Finally, a similarity score is generated for each recipe in the dataset based on the user-input ingredients list and the recipes are recommended based on highest scores. 


## Objective

- Given any recipe dataset, build a model to find the most relevant recipes that match the user’s ingredients list.
- Perform unsupervised Topic Modeling on the recipes to group recipes into topics.
- Rank the ingredients based on its frequency i.e. each successive ingredient in the list is weighted incrementally less.
- Create a search algorithm that utilizes similarity scoring to rank recipes according to the greatest similarity to the user-input list of ingredients and returns recipe recommendations based on the scores. 


## Dataset Description:

The kaggle dataset used for our recommendation system was created by scraping AllRecipes.com, a popular social network recipe site. For each recipe, the dataset contains a corresponding row with recipe id, name, average ratings of reviewers, image url, review nums, ingredients, cooking directions, nutritions, and reviews. We have pre-filtered the dataset to exclude records which contains repeated ingredients, no images or zero reviews to ensure data quality. The final dataset includes 49,698 recipes with 38,131 ingredients representing a varied range of cuisines and tastes. Given below is the description of the columns available in the dataset.

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


## Dataset Requirements for any new recipe datasets

Incase you decide to use your own dataset other than the one listed above, please make sure your dataset meets the following requirements:

| Column | Datatype | Required |
| ------ | -------- | -------- |
| recipe_name | string | Yes |
| ingredients | list of strings seperated by comma | Yes |
| cooking_directions | string | Yes |

If any of the above columns are not available or if there is a column name or data type mismatch the recommender function will fail.

## Data Preprocessing

To perform preprocessing on the data that is downloaded from the source, we will use the utils.py file. The filepath of the raw data along with the column names are passed to the utils which performs some basic preprocessing and outputs csv file which is saved as cleaned_data.csv in the data folder.

## Data Features:

- Recipe Name
- Ingredients
- Cooking Directions

## Project folder structure
<img width="427" alt="Screen Shot 2021-03-15 at 5 30 57 PM" src="https://user-images.githubusercontent.com/29209748/111242231-f8886900-85bb-11eb-908b-2b01f4219730.png">


## Installation and user guide on how to use CookSmart

To install and use CookSmart, you can follow the below steps or refer to the example section below.

1. Clone the repository:
	```git clone https://github.com/sandeeptiwari6/cooksmart```
2. 
	```cd cooksmart```
3. 
	```python setup.py install```


## Example
   
   ```
   from cooksmart import RecipeRecommender
   
   r = RecipeRecommender()
   r.fit()
   query = ["pepper", "chicken", "pesto", "vinegar", "tomato", "cheese"]
   r.get_recommendations(query)
   r.visualize_fit()
   ```
  Simply change your query list of ingredients. The end result is a table of recipe recommendations that has the recipe name, ingredients and cooking directions.


## Limitations

In its current form, the recommender system does not handle recommendations based on priority of the input ingredients which might produce undesired results. The util.py file handles basic data preprocessing and can be expanded further to accomadate advance functionalities to handle various recipe datasets outside our package. The package does not have an UI interface in its current state, but providing one in the future would make the interaction seamless for non-technical users. 

