from typing import Iterable

from sklearn.feature_extraction.text import CountVectorizer


class BagOfWordsModel:
    """
    A Bag-of-Words representation for a text corpus.

    # Attributes
    * matrix (scipy.sparse.csr_matrix): The document-term matrix of word counts.
    * feature_names (list[str]): List of feature names corresponding to the matrix columns.
    *
    # Usage
    ```
    bow = BagOfWordsModel(df["lemmatized_str"])
    ```
    """

    def __init__(
        self,
        texts: Iterable[str],
        min_freq: int | float | None = None,
    ):
        """
        Initialize the BagOfWordsModel by fitting the vectorizer to the text corpus. This also filters out tokens
        that do not appear more than five times in the dataset.

        This sets its tokenizer to the word boundary tokenizer since the input, at this point, **should** be
        cleaned and processed text.

        This also uses both unigrams and bigrams, hence, at the worst case its space complexity is O(n^2).

        # Parameters
        * texts: An iterable of cleaned text documents.
        * min_freq: Determines the document frequency of a token for it to appear in the model.
        Can be a type of int (i.e., the token must appear min_freq number of times in the document)
        or a float (i.e, token must be in min_freq% of the documents)
        """
        vectorizer = CountVectorizer(
            min_df=min_freq if min_freq is not None else 1,
            tokenizer=str.split,  # Use str.split instead of lambda
            lowercase=False,  # Don't lowercase
            ngram_range=(1, 2),  # Unigrams and bigrams
        )
        self.matrix = vectorizer.fit_transform(texts)
        self.feature_names = vectorizer.get_feature_names_out()
        self.vectorizer = vectorizer
        self.sparsity = self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1])

    def transform_sentence(self, sentence: str):
        """
        Returns the embedding of the sentence using the BoW matrix.

        # Parameters:
        * sentence: Cleaned sentence to vectorize.

        # Returns
        Sentence embedding.
        """
        return self.vectorizer.transform([sentence])
