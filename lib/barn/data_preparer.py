from typing import Optional, Dict, Union, Any
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from scipy.sparse import spmatrix, csr_matrix


class TorchableSet(Dataset[tuple[Tensor, Tensor]]):
    """
    A PyTorch Dataset wrapper for sparse feature matrices and label arrays.

    Converts scipy sparse matrices and numpy/pandas labels into a format suitable for PyTorch DataLoaders.
    Labels are mapped to contiguous integers (default: {-1: 0, 0: 1, 1: 2} for sentiment analysis).

    # Parameters
    * X: Sparse feature matrix (e.g., TF-IDF or bag-of-words).
    * y: Label array as numpy ndarray or pandas Series.
    * mapping: Optional dictionary to map original labels to contiguous integers.
    * validate: Whether to validate that X and y have matching row counts.

    # Attributes
    * X: The sparse feature matrix as csr_matrix.
    * y: The mapped labels as numpy ndarray.
    * mapping: The label mapping dictionary.
    * vocab_size: Number of features (columns in X).
    """

    def __init__(
        self,
        X: spmatrix,
        y: Union[np.ndarray, Any],
        mapping: Optional[Dict[int, int]] = None,
        validate: bool = True,
    ) -> None:
        self.mapping: Dict[int, int] = mapping or {-1: 0, 0: 1, 1: 2}

        if hasattr(y, "map"):
            y = y.map(self.mapping)  # type: ignore[union-attr]
        else:
            y = np.vectorize(self.mapping.get)(y)

        self.y: np.ndarray = (
            y.values if hasattr(y, "values") else np.asarray(y)  # type: ignore[union-attr]
        )

        self.X: csr_matrix = csr_matrix(X)

        if validate:
            self._validate()

    def _validate(self) -> None:
        """
        Validate that the feature matrix and labels have matching dimensions.

        # Raises
        AssertionError if X.shape[0] != y.shape[0].
        """
        assert self.X.shape[0] == self.y.shape[0]  # type: ignore[index]

    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size (number of features).

        # Returns
        Number of columns in the feature matrix.
        """
        return self.X.shape[1]  # type: ignore[index]

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        # Returns
        Number of rows in the feature matrix.
        """
        return self.X.shape[0]  # type: ignore[index]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Get a single sample by index.

        # Parameters
        * idx: Index of the sample to retrieve.

        # Returns
        Tuple of (features, label) as PyTorch tensors.
        """
        x: Tensor = torch.from_numpy(
            self.X[idx].toarray().ravel()
        ).float()

        y: Tensor = torch.tensor(self.y[idx], dtype=torch.long)

        return x, y
