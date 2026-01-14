import re
import unicodedata
import string

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
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')

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
    return re.sub(f" +", " ", text).strip()


def rem_stopwords(tokens: list[str], stopwords: set[str]) -> str:
    """
    This removes all stopwords. For fast look up, `stopwords` is a set with type str.
    This allows for constant time lookups as opposed to a linear search with a list.

    Strings detected as stopwords is replaced with the empty string "".

    Do not use this function alone, use `clean_and_tokenize()`.

    # Parameters
    * text: A tokenized string. 
    * stopwords: stopword dictionary.
    
    # Returns
    Text with the stopwords removed.

    # Examples
    rem_stopwords(["he", "is", "an", "amazing", "master",], stopwords_lut)
    $ ['amazing', 'master']

    # Future
    If this function is too slow, we may implement `stopwords` as an 26-ary trie to achieve log-linear time.
    """
    filtered = [word for word in tokens if word not in stopwords]
    return filtered

    
def clean_and_tokenize(text: str, stopwords: set[str]) -> list[str]:
    """
    This is the main function for data cleaning (i.e., it calls all the cleaning functions in the prescribed order).

    This function should be used as a first-class function in a map.

    # Parameters
    * text: The string entry from a DataFrame column.
    * stopwords: stopword dictionary.

    # Returns
    Clean tokenized string. 
    """
    # cleaning on the base string
    text = normalize(text)
    text = rem_punctuation(text)
    text = rem_numbers(text)
    text = collapse_whitespace(text)
    
    # tokenization is now trivial since the cleaning steps allow the tokenization to be a mere word boundary split
    toks = text.split()

    # cleaning on the tokenized string
    toks = rem_stopwords(toks, stopwords)

    return toks


def find_spam_and_empty(tokens: list[str], min_length: int = 3) -> list[str]:
    """
    Filter out empty token lists and unintelligible/spammy tokens.
    
    Spammy tokens:
    - Tokens shorter than min_length
    - Tokens containing non-alphabetic characters
    - Tokens consisting of a repeated substring (e.g., 'aaaaaa', 'ababab', 'abcabcabc')
    
    # Parameters
    * tokens: list of token strings
    * min_length: minimum length of token to keep
    
    # Returns
        Cleaned list of tokens, or None if empty after filtering
    """
    cleaned = []
    for t in tokens:
        if len(t) < min_length:  
            continue
        if not t.isalpha():
            continue
        if re.fullmatch(r'(.+?)\1+', t):
            continue
        cleaned.append(t)
    return cleaned