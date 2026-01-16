import nltk


nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.stem import WordNetLemmatizer

wn_lemmatizer = WordNetLemmatizer()


def lemmatizer(text: str) -> str:
    """
    Lemmatize a string of space-separated tokens and return the lemmatized string.

    # Parameters
    * text: A string of space-separated tokens.

    # Returns
    A string with lemmatized tokens.
    """
    tokens = text.split() if text else []
    lemmatized = [wn_lemmatizer.lemmatize(t) for t in tokens]
    lemmatized_str = " ".join(lemmatized)

    return lemmatized_str
