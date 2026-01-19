import spacy

# STATE REQUIREMENT // SIDE-EFFECT
# Requires:
#   uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
nlp = spacy.load("en_core_web_sm")


def lemmatizer(text: str) -> str:
    """
    Lemmatize a string of space-separated tokens and return the lemmatized string.

    # Parameters
    * text: A string of space-separated tokens.

    # Returns
    A string with lemmatized tokens.
    """

    if not text:
        return ""
    doc = nlp(text)

    return " ".join(
        token.lemma_ for token in doc if not token.is_space and not token.is_punct
    )
