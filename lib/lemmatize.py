import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()


def lemmatize(tokens: list[str]) -> pd.Series:
    """
    This returns a Series object containing the lemmatized version
    of the tokenized string.

    # Params
    * tokens: the tokenized string.

    # Returns
    Series object of the lemmatized token string.
    """
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    lemmatized_str = " ".join(lemmatized)

    return pd.Series([lemmatized, lemmatized_str])
