class DataFormatError(Exception):
    """
    Exception Error if dataframe is in
    wrong format
    """
    def __init__(self, salary, message="Invalid data"):
        super().__init__(message)


class QueryError(Exception):
    """
    Exception error if inputted query is invalid
    """
    def __init__(self, salary, message="Invalid Query"):
        super().__init__(message)
