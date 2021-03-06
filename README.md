## Data 515: Software Design for Data Science

## Introduction 

Many factors influence the meals being cooked at home today. People are always on the lookout for inspiration to cook something new provided they have the time, interest, creativity and the ingredients at hand. Knowing what different recipes one can cook given a list of ingredients is one of the constant and colossal challenges in life. The goal of this project is to use the recipe dataset available in kaggle to build a recipe recommendation system that allows users to search for recipes based on the ingredients list. The recommendation system will perform unsupervised topic modeling on the recipes by vectorizing the recipe name, ingredients and user-input ingredients list using TF-IDF. The vectorized representations are then provided as an input to the LDA model to generate a topic probability distribution. Finally, a similarity score is generated for each recipe in the dataset based on the user-input ingredients list and the recipes are recommended based on highest scores. 

## Dataset Description:

The kaggle dataset used for our recommendation system was created by scraping AllRecipes.com, a popular social network recipe site. With over 49,085 recipes, the dataset represents a varied range of cuisines and tastes. Given below is the description of the columns available in the dataset used for this project.

| Column | Datatype | Value |
| ------ | -------- | ----- |
| recipe_id | int | A unique id associated with each recipe |

## Data Processing:

## Objective

- Given any recipe dataset, build a model to find most relevant recipes that match the user’s ingredients list.
- Perform unsupervised Topic Modeling on the recipes to group recipes into topics.
- Rank the ingredients based on its frequency i.e. each successive ingredient in the list is weighted incrementally less.
- Create a search algorithm that utilizes similarity scoring to rank recipes according to the greatest similarity to the user-input list of ingredients and returns recipe recommendations based on the scores. 





## Installation & packages and their versions


## License Information

