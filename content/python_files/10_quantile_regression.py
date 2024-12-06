# %% [markdown]
#
# # Quantile regression
#
# This notebook explores how to predict intervals with available techniques in
# scikit-learn.
#
# We cover a subset of available techniques. For instance, conformal predictions
# handle this specific task - see packages like MAPIE for broader coverage:
# https://github.com/scikit-learn-contrib/MAPIE.
#
# ## Predicting intervals with linear models
#
# This section revisits linear models and shows how to predict intervals with
# quantile regression.
#
# First, let's load our penguins dataset for our regression task.

# %%
# When using JupyterLite, uncomment and install the `skrub` package.
# %pip install skrub
import matplotlib.pyplot as plt
import skrub

skrub.patch_display()

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_regression.csv")
penguins

# %% [markdown]
#
# In this dataset, we predict the body mass of a penguin given its flipper
# length.

# %%
X = penguins[["Flipper Length (mm)"]]
y = penguins["Body Mass (g)"]

# %% [markdown]
#
# In our study of linear models, we saw that `LinearRegression` minimizes the mean
# squared error and predicts the conditional mean of the target.
#
# Here, we fit this model and predict several data points between the minimum and
# maximum flipper length.

# %%
from sklearn.linear_model import LinearRegression

model_estimate_mean = LinearRegression()
model_estimate_mean.fit(X, y)

# %%
import numpy as np

X_test = pd.DataFrame(
    {"Flipper Length (mm)": np.linspace(X.min(axis=None), X.max(axis=None), 100)}
)
y_pred_mean = model_estimate_mean.predict(X_test)

# %%
_, ax = plt.subplots()
penguins.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)", ax=ax, alpha=0.5)
ax.plot(
    X_test["Flipper Length (mm)"],
    y_pred_mean,
    color="tab:orange",
    label="predicted mean",
    linewidth=3,
)
ax.legend()
plt.show()

# %% [markdown]
#
# We discussed how mean estimators become sensitive to outliers. Sometimes we
# prefer a more robust estimator like the median.
#
# Here, `QuantileRegressor` minimizes the mean absolute error and predicts the
# conditional median.

# %%
from sklearn.linear_model import QuantileRegressor

model_estimate_median = QuantileRegressor(quantile=0.5)
model_estimate_median.fit(X, y)
y_pred_median = model_estimate_median.predict(X_test)

# %%
_, ax = plt.subplots()
penguins.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)", ax=ax, alpha=0.5)
ax.plot(
    X_test["Flipper Length (mm)"],
    y_pred_mean,
    color="tab:orange",
    label="predicted mean",
    linewidth=3,
)
ax.plot(
    X_test["Flipper Length (mm)"],
    y_pred_median,
    color="tab:green",
    label="predicted median",
    linewidth=3,
    linestyle="--",
)
ax.legend()
plt.show()

# %% [markdown]
#
# For confidence intervals, we want to predict specific quantiles. We generalize
# quantile regression beyond the median. The pinball loss generalizes the mean
# absolute error for any quantile.
#
# The `quantile` parameter sets which quantile to predict. For an 80% prediction
# interval, we predict the 10th and 90th percentiles.

# %%
model_estimate_10 = QuantileRegressor(quantile=0.1)
model_estimate_90 = QuantileRegressor(quantile=0.9)

model_estimate_10.fit(X, y)
model_estimate_90.fit(X, y)

y_pred_10 = model_estimate_10.predict(X_test)
y_pred_90 = model_estimate_90.predict(X_test)

# %%
_, ax = plt.subplots()
penguins.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)", ax=ax, alpha=0.5)
ax.plot(
    X_test["Flipper Length (mm)"],
    y_pred_mean,
    color="tab:orange",
    label="predicted mean",
    linewidth=3,
)
ax.plot(
    X_test["Flipper Length (mm)"],
    y_pred_median,
    color="tab:green",
    label="predicted median",
    linewidth=3,
    linestyle="--",
)
ax.fill_between(
    X_test["Flipper Length (mm)"],
    y_pred_10,
    y_pred_90,
    alpha=0.2,
    label="80% coverage interval",
)
ax.legend()
plt.show()

# %% [markdown]
#
# ## Predicting intervals with tree-based models
#
# **Exercise**:
#
# Now repeat the previous experiment using `HistGradientBoostingRegressor`. Read
# the documentation to find the parameters that optimize the right loss function.
#
# Plot the conditional mean, median and 80% prediction interval.

# %%
# Write your code here.
