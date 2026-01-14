import pandas as pd
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def lemmatize_toks(text: str) -> pd.Series:
    """
    Lemmatize a string of space-separated tokens and return a Series containing
    the lemmatized string.

    # Parameters
    * text: A string of space-separated tokens.

    # Returns
    A pd.Series containing a single string with lemmatized tokens.
    """
    tokens = text.split() if text else []
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    lemmatized_str = " ".join(lemmatized)

    return pd.Series([lemmatized_str])
