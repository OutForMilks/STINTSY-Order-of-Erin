import nltk
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def stem_and_lemmatize(tokens: list[str]) -> pd.Series:
    """
    This returns a Series object containing the stemmed and lemmatized version
    of the tokenized string.

    # Params
    * tokens: the tokenized string.

    # Returns
    Series object of the stemmed and lemmatized token string.
    """
    stemmed = [stemmer.stem(t) for t in tokens]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    lemmatized_str = " ".join(lemmatized)

    return pd.Series([stemmed, lemmatized, lemmatized_str])
