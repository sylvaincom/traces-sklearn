# %% [markdown]
#
# # Linear models - Losses
#
# This notebook explores linear models and loss functions in depth. We use the previous
# regression problem that models the relationship between penguins' flipper length and
# body mass.

# %%
# When using JupyterLite, you will need to uncomment and install the `skrub` package.
# %pip install skrub
import matplotlib.pyplot as plt
import skrub
skrub.patch_display()  # make nice display for pandas tables

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_regression.csv")
data

# %%
data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
plt.show()

# %% [markdown]
#
# The data shows a clear linear relationship between flipper length and body mass. We
# use body mass as our target variable and flipper length as our feature.

# %%
X, y = data[["Flipper Length (mm)"]], data["Body Mass (g)"]

# %% [markdown]
#
# In the previous notebook, we used scikit-learn's `LinearRegression` to learn model
# parameters from data with `fit` and make predictions with `predict`.

# %%
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
predicted_target = model.predict(X)

# %%
ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.plot(X, predicted_target, label=model.__class__.__name__, color="tab:orange", linewidth=4)
ax.legend()
plt.show()
# %% [markdown]
#
# The linear regression model minimizes the error between true and predicted targets.
# A general term to describe this error is "loss function". For the linear regression,
# from scikit-learn, it specifically minimizes the least squared error:
#
# $$
# loss = (y - \hat{y})^2
# $$
#
# or equivalently:
#
# $$
# loss = (y - X \beta)^2
# $$
#
# Let's visualize this loss function:

# %%
def se_loss(true_target, predicted_target):
    loss = (true_target - predicted_target) ** 2
    return loss


# %%
import numpy as np

xmin, xmax = -2, 2
xx = np.linspace(xmin, xmax, 100)

# %%
plt.plot(xx, se_loss(0, xx), label="SE loss")
plt.legend()
plt.show()

# %% [markdown]
#
# The bell shape of the loss function heavily penalizes large errors, which
# significantly impacts the model fit.

# %% [markdown]
#
# **EXERCISE**
#
# 1. Add an outlier to the dataset: a penguin with 230 mm flipper length and 300 g
#    body mass
# 2. Plot the updated dataset
# 3. Fit a `LinearRegression` model on this dataset, using `sample_weight`
#    to give the outlier 10x more weight than other samples
# 4. Plot the model predictions
#
# How does the outlier affect the model?

# %%
# Write your code here.

# %% [markdown]
#
# Instead of squared loss, we now use the Huber loss through scikit-learn's
# `HuberRegressor`. We fit this model similarly to our previous approach.

# %%
from sklearn.linear_model import HuberRegressor

sample_weight = np.ones_like(y)
sample_weight[-1] = 10
model = HuberRegressor()
model.fit(X, y, sample_weight=sample_weight)
predicted_target = model.predict(X)
# -

ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.plot(X, predicted_target, label=model.__class__.__name__, color="black", linewidth=4)
plt.legend()
plt.show()

# %% [markdown]
#
# The Huber loss gives less weight to outliers compared to least squares.

# %% [markdown]
#
# **EXERCISE**
#
# 1. Read the `HuberRegressor` documentation
# 2. Create a `huber_loss` function similar to `se_loss`
# 3. Create an absolute loss function
#
# Explain why outliers affect Huber regression less than ordinary least squares.

# %%
# Write your code here.

# %% [markdown]
#
# Huber and absolute losses penalize outliers less severely. This makes outliers less
# influential when finding the optimal $\beta$ parameters. The `HuberRegressor`
# estimates the median rather than the mean.
#
# For other quantiles, scikit-learn offers the `QuantileRegressor`. It minimizes the
# pinball loss to estimate specific quantiles. Here's how to estimate the median:

# %%
from sklearn.linear_model import QuantileRegressor

model = QuantileRegressor(quantile=0.5)
model.fit(X, y, sample_weight=sample_weight)
predicted_target = model.predict(X)


# %%
ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.plot(X, predicted_target, label=model.__class__.__name__, color="black", linewidth=4)
ax.legend()
plt.show()

# %% [markdown]
#
# The `QuantileRegressor` enables estimation of confidence intervals:

# %%
model = QuantileRegressor(quantile=0.5, solver="highs")
model.fit(X, y, sample_weight=sample_weight)
predicted_target_median = model.predict(X)

model.set_params(quantile=0.90)
model.fit(X, y, sample_weight=sample_weight)
predicted_target_90 = model.predict(X)

model.set_params(quantile=0.10)
model.fit(X, y, sample_weight=sample_weight)
predicted_target_10 = model.predict(X)

# %%
ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.plot(
    X,
    predicted_target_median,
    label=f"{model.__class__.__name__} - median",
    color="black",
    linewidth=4,
)
ax.plot(
    X,
    predicted_target_90,
    label=f"{model.__class__.__name__} - 90th percentile",
    color="tab:orange",
    linewidth=4,
)
ax.plot(
    X,
    predicted_target_10,
    label=f"{model.__class__.__name__} - 10th percentile",
    color="tab:green",
    linewidth=4,
)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %% [markdown]
#
# This plot shows an 80% confidence interval around the median using the 10th and 90th
# percentiles.
