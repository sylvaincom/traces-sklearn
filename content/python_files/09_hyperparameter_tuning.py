# %% [markdown]
#
# # Hyperparameters tuning
#
# Previous notebooks showed how model parameters impact statistical performance. We want
# to optimize these parameters to achieve the best possible model performance. This
# optimization process is called hyperparameter tuning.
#
# This notebook demonstrates several methods to tune model hyperparameters.
#
# ## Introductory example
#
# We revisit an example from the linear models notebook about the impact of the $\alpha$
# parameter in a `Ridge` model. The $\alpha$ parameter controls model regularization
# strength. No general rule exists for selecting a good $\alpha$ value - it depends on
# the specific dataset.
#
# Let's load a dataset for regression:

# %%
# When using JupyterLite, uncomment and install the `skrub` and `pyodide-http` packages.
# %pip install skrub
# %pip install pyodide-http
import matplotlib.pyplot as plt
import skrub

# import pyodide_http
# pyodide_http.patch_all()

skrub.patch_display()  # makes nice display for pandas tables

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X

# %%
y

# %% [markdown]
#
# Now we define a `Ridge` model that processes data by adding feature interactions using
# a `PolynomialFeatures` transformer.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

model = Pipeline(
    [
        ("poly", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ]
)
model

# %% [markdown]
#
# We start with scikit-learn's default parameters. Let's evaluate this basic model:

# %%
import pandas as pd
from sklearn.model_selection import cross_validate, KFold

cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(model, X, y, cv=cv)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
cv_results.aggregate(["mean", "std"])

# %% [markdown]
#
# Nothing indicates our pipeline achieves optimal performance. The `PolynomialFeatures`
# degree might need adjustment or the `Ridge` regressor might need different
# regularization. Let's examine which parameters we could tune:

# %%
for params in model.get_params():
    print(params)

# %% [markdown]
#
# Two key parameters are `scaler__degree` and `ridge__alpha`. We will find
# their optimal values for this dataset.
#
# ## Manual hyperparameters search
#
# Before exploring scikit-learn's automated tuning tools, we implement a simplified
# manual version.
#
# **EXERCISE**:
#
# 1. Create nested `for` loops to try all parameter combinations defined in
#    `parameter_grid`
# 2. In the inner loop, use cross-validation on the training set to get an array of
#    scores
# 3. Compute the mean and standard deviation of cross-validation scores to find the best
#    hyperparameters
# 4. Train a model with the best hyperparameters and evaluate it on the test set

# %%
# Write your code here.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

parameter_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10],
}

# %% [markdown]
#
# ## Hyperparameters search using a grid
#
# Our manual search implements a grid-search: trying every possible parameter
# combination. Scikit-learn provides `GridSearchCV` to automate this process. During
# fitting, it performs cross-validation and selects optimal hyperparameters.

# %%
from sklearn.model_selection import GridSearchCV

search_cv = GridSearchCV(model, param_grid=parameter_grid)
search_cv.fit(X_train, y_train)

# %% [markdown]
#
# The `best_params_` attribute shows the optimal parameters found:

# %%
search_cv.best_params_

# %% [markdown]
#
# The `cv_results_` attribute provides details about all hyperparameter combinations
# tried during fitting:

# %%
cv_results = pd.DataFrame(search_cv.cv_results_)
cv_results

# %% [markdown]
#
# When `refit=True` (default), the search trains a final model using the best
# parameters. Access this model through `best_estimator_`:

# %%
search_cv.best_estimator_

# %% [markdown]
#
# The `best_estimator_` handles `predict` and `score` calls to `GridSearchCV`:

# %%
search_cv.score(X_test, y_test)

# %% [markdown]
#
# **EXERCISE**:
#
# `GridSearchCV` behaves like any classifier or regressor. Use `cross_validate` to
# evaluate the grid-search model we created.

# %%
# Write your code here.

# %% [markdown]
#
# **QUESTION**:
#
# What limitations does the grid-search approach have?

# %% [markdown]
#
# ## Randomized hyperparameters search
#
# Grid-search has two main limitations:
#
# - It explores only predefined parameter combinations
# - Adding parameters or values exponentially increases search cost
#
# `RandomizedSearchCV` draws parameter values from specified distributions. This allows
# non-grid exploration of the hyperparameter space with a fixed computational budget.

# %%
import numpy as np
from scipy.stats import loguniform

parameter_distributions = {
    "poly__degree": np.arange(1, 5),
    "ridge__alpha": loguniform(1, 3),
}

# %%
from sklearn.model_selection import RandomizedSearchCV

search_cv = RandomizedSearchCV(
    model,
    param_distributions=parameter_distributions,
    n_iter=10,
)

# %%
cv_results = cross_validate(search_cv, X, y, cv=cv, return_estimator=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
for est in cv_results["estimator"]:
    print(est.best_params_)

# %% [markdown]
#
# ## Model with internal hyperparameter tuning
#
# Some estimators include efficient hyperparameter selection, more efficient than
# grid-search. These estimators typically end with `CV` (e.g. `RidgeCV`).
#
# **EXERCISE**:
#
# 1. Create a pipeline with `PolynomialFeatures`, `StandardScaler`, and `Ridge`
# 2. Create a grid-search with this pipeline and tune `alpha` using `np.logspace(-2, 2,
#    num=50)`
# 3. Fit the grid-search on the training set and time it
# 4. Repeat using `RidgeCV` instead of `Ridge` and remove `GridSearchCV`
# 5. Compare computational performance between approaches

# %%
# Write your code here.

# %% [markdown]
#
# ## Inspection of hyperparameters in cross-validation
#
# When performing search cross-validation inside evaluation cross-validation, different
# hyperparameter values may emerge for each split. Let's examine this with
# `GridSearchCV`:

# %%
from sklearn.linear_model import RidgeCV

inner_model = Pipeline(
    [
        ("poly", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ]
)
param_grid = {"poly__degree": [1, 2], "ridge__alpha": np.logspace(-2, 2, num=10)}
model = GridSearchCV(inner_model, param_grid=param_grid, n_jobs=-1)
model

# %% [markdown]
#
# We run cross-validation and store models from each split by setting
# `return_estimator=True`:

# %%
cv_results = cross_validate(model, X, y, cv=cv, return_estimator=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %% [markdown]
#
# The `estimator` column contains the different estimators. We examine `best_params_`
# from each `GridSearchCV`:

# %%
for estimator_cv_fold in cv_results["estimator"]:
    print(estimator_cv_fold.best_params_)

# %% [markdown]
#
# This inspection reveals the stability of hyperparameter values across folds.
