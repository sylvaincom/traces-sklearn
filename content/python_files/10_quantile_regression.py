# %% [markdown]
#
# # Quantile regression
#
# In this notebook, we go into more details on how to predict intervals with the
# available techniques in scikit-learn.
#
# Note that we will go only in a subset of available techniques. For instance, conformal
# predictions are design for this specific task and you can have a look at packages as
# MAPIE to have a broader coverage: https://github.com/scikit-learn-contrib/MAPIE.
#
# ## Predicting intervals with linear models
#
# In this section, we come back to linear models and recall a way to predict intervals
# with quantile regression.
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
# In this dataset, the goal is to predict the body mass of a penguin given its flipper
# length.

# %%
X = penguins[["Flipper Length (mm)"]]
y = penguins["Body Mass (g)"]

# %% [markdown]
#
# When we studied linear models, we saw that the `LinearRegression` is an estimator that
# minimizes the mean squared error and thus predicts the conditional mean of the target.
#
# Here, let's fit this model and predict severals data points between the minimum and
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
# We discussed that the mean estimator could be sensitive to outliers and that sometimes
# we might prefer to get a more robust estimator such as the median.
#
# In this case, we can use the `QuantileRegressor` that minimizes the mean absolute
# error and thus predicts the conditional median.

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
# When it comes to confidence intervals, we might be interested to get a prediction of
# some given quantiles. So we would like to generalize the quantile regression to get
# other quantiles than the median. Fortunately, the pinball loss is a generalization of
# the mean absolute error for any quantile.
#
# Indeed, using the `quantile` parameter, we can set the quantile that we want to
# predict. So if we are interested to get a 80% prediction interval, we can predict the
# 10th and 90th percentiles.

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
# Now, this is your turn to repeat the previous experiment using the
# `HistGradientBoostingRegressor`. You will have to read the documentation and check
# how to change the parameters to optimize the right loss function.
#
# Plot the conditional mean, median and 80% prediction interval.

# %%
# Write your code here.
