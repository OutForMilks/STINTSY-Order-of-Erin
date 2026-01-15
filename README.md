# Sentiment Analysis Project on Twitter Data.

**By**: Stephen Borja, Justin Ching, Erin Chua, and Zhean Ganituen.

**Dataset**: Hussein, S. (2021). Twitter Sentiments Dataset [Dataset]. Mendeley. https://doi.org/10.17632/Z9ZW7NT5H2.1

**Motivation**: Every minute, social media users generate a large influx of textual data on live events. Performing sentiment analysis on this data provides a real-time view of public perception, enabling quick insights into the general population’s opinions and reactions.

**Goal**: By the end of the project, our goal is to create and compare supervised learning algorithms for sentiment analysis.

## Set Up

> This project requires [uv](https://docs.astral.sh/uv/getting-started/installation/) as its package manager. The remaining dependencies are handled by uv itself (you can view these dependencies in [`pyproject.toml`](/pyproject.toml)).

1. Simply clone the repository.
2. Run `uv sync` which should catch your environment up.
3. If your text editor does not support JupyterNotebooks run browser-based interface with `uv run --with jupyter jupyter lab`. No need to do this if you're using PyCharm or VSCode.
4. Run `uv run scripts/codegen.py` and place the result in the `notebooks/` directory, this script generates some prerequisites the Jupyter notebooks require.

## Machine Learning Models

We will use two classical supervised learning algorithms: **Naive Baye** and 🏗️. Then, one neural network approach: 🏗️.

> @Stephen: add the classical supervised learning algorithm and neural network approach here.
> When you add the algorithm update: 
> - the ## Notebooks section in this file
> - the filename of `notebooks/svm.ipynb`

## Notebooks

View the notebooks in the `notebooks/` directory. The notebooks are enumarated below, also view the notebooks in this order indicated.

1. `data.ipynb`: contains data description, data cleaning, exploratory data analysis, and preprocessing.
1. `bayes.ipynb`: contains model training, error analysis, and finetuning for Naive Bayes.
1. `svm.ipynb`: contains model training, error analysis, and finetuning for SVM.
1. `nn.ipynb`: contains model training, error analysis, and finetuning for the Neural Network approach.
1. `eval.ipynb`: contains the model evaluation and conclusion of the project.

## Data

The _Twitter Sentiments Dataset_ used in the project is from Hussein (2021). The dataset is freely available in Mendeley Data via
[doi: 10.17632/z9zw7nt5h2.1](https://doi.org/10.17632/Z9ZW7NT5H2.1) under the CreativeCommons license CC BY 4.0.

The dataset is also available as a `.csv` in the repository, please see the `data/` directory.
