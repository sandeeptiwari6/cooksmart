def is_valid_recipe_df(data):
    """
    checks to see if column are valid
    """
    for col_name in ['recipe_name', 'ingredients',
                     'cooking_directions']:
        if col_name not in data.columns:
            return False
    return True
