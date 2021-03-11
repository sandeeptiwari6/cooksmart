class DataFormatError(Exception):
    def __init__(self, salary, message="Invalid data"):
        super().__init__(message)


class QueryError(Exception):
    def __init__(self, salary, message="Invalid Query"):
        super().__init__(message)


def is_valid_recipe_df(data):
    for col_name in ['recipe_name', 'ingredients', 'cooking_directions']:
        if col_name not in data.columns:
            return False
    # check type for ingredients
    return True
