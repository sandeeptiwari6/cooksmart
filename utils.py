import pandas as pd


def ingredients_to_text(ingredient_list):
    return " ".join(eval(ingredient_list))


def preprocess(filename, toFile='cleaned_data.csv',
               ingredients_column='ingredients', name_column='recipe_name',
               direction_column='cooking_directions', delimiter=','):
    """

    :param filename: file to recipes dataset
    :param delimiter: character separating ingredients
    :return: cleaned dataframe
    """
    data = pd.read_csv(filename)

    data[ingredients_column] = data[ingredients_column].str.lower()
    data[ingredients_column] = data[ingredients_column].apply(
                                        ingredients_to_text)
    data.reset_index(inplace=True)
    data.to_csv(toFile)
