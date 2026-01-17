import re
import string
import unicodedata


def normalize(text: str) -> str:
    """
    Normalize text from a pandas entry to ASCII-only lowercase characters. Hence, this removes Unicode characters with no ASCII
    equivalent (e.g., emojis and CJKs).

    Do not use this function alone, use `clean_and_tokenize()`.

    # Parameters
    * text: String entry.

    # Returns
    ASCII-normalized text containing only lowercase letters.

    # Examples
    normalize("¿Cómo estás?")
    $ 'como estas?'

    normalize(" hahahaha HUY! Kamusta 😅 Mayaman $$$ ka na ba?")
    $ ' hahahaha huy! kamusta  mayaman $$$ ka na ba?'
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")

    return ascii_text.lower()


def rem_punctuation(text: str) -> str:
    """
    Removes the punctuations. This function simply replaces all punctuation marks and special characters
    to the empty string. Hence, for symbols enclosed by whitespace, the whitespace are not collapsed to a single whitespace
    (for more information, see the examples).

    Do not use this function alone, use `clean_and_tokenize()`.

    # Parameters
    * text: String entry.

    # Returns
    Text with the punctuation removed.

    # Examples
    rem_punctuation("this word $$ has two spaces after it!")
    $ 'this word  has two spaces after it'

    rem_punctuation("these!words@have$no%space")
    $ 'thesewordshavenospace'
    """
    return re.sub(f"[{re.escape(string.punctuation)}]", "", text)


def rem_numbers(text: str) -> str:
    """
    Removes numbers. This function simply replaces all numerical symbols to the empty string. Hence, for symbols enclosed by
    whitespace, the whitespace are not collapsed to a single whitespace (for more information, see the examples).

    Do not use this function alone, use `clean_and_tokenize()`.

    # Parameters
    * text: String entry.

    # Returns
    Text with the numerical symbol removed

    # Examples
    rem_numbers(" h3llo, k4must4 k4  n4?")
    ' hllo, kmust k  n?'
    """
    return re.sub(r"\d+", "", text)


def collapse_whitespace(text: str) -> str:
    """
    This collapses whitespace. Here, collapsing means the transduction of all whitespace strings of any
    length to a whitespace string of unit length (e.g., "   " -> " "; formally " "+ -> " ").

    Do not use this function alone, use `clean_and_tokenize()`.

    # Parameters
    * text: String entry.

    # Returns
    Text with the whitespaces collapsed.

    # Examples
    collapse_whitespace("  huh,  was.  that!!! ")
    $ 'huh, was. that!!!'
    """
    return re.sub(" +", " ", text).strip()


def rem_stopwords(text: str, stopwords: set[str]) -> str:
    """
    This removes all stopwords. For fast look up, `stopwords` is a set with type str.
    This allows for constant time lookups as opposed to a linear search with a list.

    Strings detected as stopwords is replaced with the empty string "".

    This assumes `text` is already lematized.

    # Parameters
    * text: A string.
    * stopwords: stopword dictionary.

    # Returns
    Text with the stopwords removed.

    # Examples
    rem_stopwords(["he", "is", "an", "amazing", "master",], stopwords_lut)
    $ ['amazing', 'master']

    # Future
    If this function is too slow, we may implement `stopwords` as an 26-ary trie to achieve log-linear time.
    """
    filtered = [word for word in text.split(" ") if word not in stopwords]

    return " ".join(filtered)


def clean(text: str) -> str:
    """
    This is the main function for data cleaning (i.e., it calls all the cleaning functions in the prescribed order).

    This function should be used as a first-class function in a map.

    # Parameters
    * text: The string entry from a DataFrame column.
    * stopwords: stopword dictionary.

    # Returns
    Clean string
    """
    # cleaning on the base string
    text = normalize(text)
    text = rem_punctuation(text)
    text = rem_numbers(text)
    text = collapse_whitespace(text)

    return text


def find_spam_and_empty(text: str, min_length: int = 3) -> str | None:
    """
    Filter out empty text and unintelligible/spammy unintelligible substrings in the text.

    Spammy substrings:
    - Shorter than min_length
    - Containing non-alphabetic characters
    - Consisting of a repeated substring (e.g., 'aaaaaa', 'ababab', 'abcabcabc')

    # Parameters
    * text: input string.
    * min_length: minimum length of word to keep.

    # Returns
        Cleaned string, or None if empty after filtering.
    """
    cleaned_tokens = []
    for t in text.split():
        if len(t) < min_length:
            continue

        if re.search(r"(.)\1{2,}", t):
            continue

        min_diversity = 0.3 + (0.1 * min(len(t), 10) / 10)
        if len(set(t)) / len(t) < min_diversity:
            continue

        if re.match(r"^(.+)\1+", t):
            continue

        cleaned_tokens.append(t)

    return " ".join(cleaned_tokens) if cleaned_tokens else None


def is_spam_token(t: str, min_length: int = 3) -> bool:
    """
    Check if a single token (word) is considered spammy/unintelligible.

    Spammy substrings:
    - Shorter than min_length
    - Containing non-alphabetic characters
    - Consisting of a repeated substring (e.g., 'aaaaaa', 'ababab', 'abcabcabc')

    # Parameters:
    * t (str): The token to check.
    * min_length (int): Minimum length for a token to be considered valid.

    Returns:
        bool: True if the token is spammy, False otherwise.
    """

    if len(t) < min_length:
        return True

    if re.search(r"(.)\1{2,}", t):
        return True

    min_diversity = 0.3 + (0.1 * min(len(t), 10) / 10)
    if len(set(t)) / len(t) < min_diversity:
        return True

    if re.match(r"^(.+)\1+", t):
        return True

    return False


def spam_affected(text: str) -> bool:
    """
    Check if a text contains at least one spammy token.

    Parameters:
    * text (str): Input text to check.

    Returns:
        bool: True if any token in text is spammy, False otherwise.
    """
    tokens = text.split()
    return any(is_spam_token(t) for t in tokens)
