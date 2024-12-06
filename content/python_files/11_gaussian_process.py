# %% [markdown]
#
# # Gaussian process
#
# All models we have encountered so far provide a point-estimate during prediction.
# However, none of these models (except `QuantileRegression`) provide confidence
# intervals for their predictions.
#
# Gaussian Process models allow us to obtain such information. In this notebook, we
# present how these models differ from the ones we already covered.
#
# Let's start by generating a toy dataset.

# %%
# When using JupyterLite, uncomment and install the `skrub` package.
# %pip install skrub
import matplotlib.pyplot as plt
import skrub

skrub.patch_display()  # makes nice display for pandas tables

# %%
import numpy as np

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))

# %%
_, ax = plt.subplots()
ax.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
ax.legend()
ax.set(xlabel="$x$", ylabel="$f(x)$", title="True generative process")
plt.show()

# %% [markdown]
#
# ## Example with noise-free target
#
# In this first example, we use the true generative process without adding noise.
# For training the Gaussian Process regression, we select only a few samples.

# %%
rng = np.random.default_rng(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

# %% [markdown]
#
# A Gaussian kernel lets us craft a kernel by hand and compose base kernels together.
# Here, we use a radial basis function (RBF) kernel and a constant parameter to fit
# the amplitude.

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

# %% [markdown]
#
# Previous methods we presented used a single model where we find optimal parameters
# that best fit the dataset. Gaussian Process uses a different paradigm: it works with
# a distribution of models. We start with a prior distribution of models. The training
# set combines with this prior to give us a posterior distribution of models.
#
# First, let's examine the prior distribution of our Gaussian process.

# %%
y_samples = gaussian_process.sample_y(X, n_samples=5)

_, ax = plt.subplots()
for idx, single_prior in enumerate(y_samples.T):
    ax.plot(
        X.ravel(),
        single_prior,
        linestyle="--",
        alpha=0.7,
        label=f"Sampled function #{idx + 1}",
    )

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(xlabel="$x$", ylabel="$f(x)$", title="Sample from the GP prior distribution")
plt.show()

# %% [markdown]
#
# The samples from the prior distribution start as random realizations. They differ
# greatly from our true generative model. However, these samples form a distribution
# of models. We plot the mean and the 95% confidence interval.

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# %%
_, ax = plt.subplots()
ax.plot(X, mean_prediction, label="Mean prediction")
ax.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.2,
    label=r"95% confidence interval",
    color="tab:blue",
)

for idx, single_prior in enumerate(y_samples.T):
    ax.plot(
        X.ravel(),
        single_prior,
        linestyle="--",
        alpha=0.7,
        label=f"Sampled function #{idx + 1}",
    )


ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(
    xlabel="$x$", ylabel="$f(x)$", title="GP prediction using only prior distribution"
)
plt.show()

# %% [markdown]
#
# The true generative process and the prediction show we need to improve our model.

# %%
_, ax = plt.subplots()
ax.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
ax.plot(X, mean_prediction, label="Mean prediction")
ax.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
    color="tab:orange",
)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(xlabel="$x$", ylabel="$f(x)$")
plt.show()

# %% [markdown]
#
# Now, we fit a Gaussian process on these few training samples to see how they
# influence the posterior distribution.

# %%
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

# %% [markdown]
#
# After fitting our model, the hyperparameters of the kernel have been optimized.
# Now, we use our kernel to compute the mean prediction of the full dataset and
# plot the 95% confidence interval.

# %%
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

_, ax = plt.subplots()
ax.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
ax.scatter(X_train, y_train, label="Observations")
ax.plot(X, mean_prediction, label="Mean prediction")
ax.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(
    xlabel="$x$",
    ylabel="$f(x)$",
    title="Gaussian process regression \non noise-free dataset",
)
plt.show()

# %% [markdown]
#
# For predictions near training points, the 95% confidence interval shows small
# amplitude. When samples fall far from training data, our model predicts less
# accurately with higher uncertainty.
#
# ## Example with noisy targets
#
# We repeat a similar experiment by adding noise to the target. This shows the
# effect of noise on the fitted model.
#
# We add random Gaussian noise to the target with an arbitrary standard deviation.

# %%
dy = 0.5 + 1.0 * rng.uniform(size=y_train.shape)
y_train_noisy = y_train + rng.normal(0, dy)

# %% [markdown]
#
# We create a similar Gaussian process model. Along with the kernel, we specify
# the parameter `alpha` which represents the variance of Gaussian noise.

# %%
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=dy**2, n_restarts_optimizer=9
)
gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# %% [markdown]
#
# Let's plot the mean prediction and uncertainty region.

# %%
_, ax = plt.subplots()
ax.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
ax.errorbar(
    X_train,
    y_train_noisy,
    dy,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
ax.plot(X, mean_prediction, label="Mean prediction")
ax.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(
    xlabel="$x$",
    ylabel="$f(x)$",
    title="Gaussian process regression \non a noisy dataset",
)
plt.show()

# %% [markdown]
#
# ## Exercise: Design a kernel for Gaussian Process
#
# ### Build the dataset
#
# The Mauna Loa Observatory collects air samples. We want to estimate the CO2
# concentration and extrapolate it for future years. Let's load the original
# dataset from OpenML.

# %%
# Uncomment this line in JupyterLite
# %pip install pyodide-http
# import pyodide_http
# pyodide_http.patch_all()

# %%
from sklearn.datasets import fetch_openml

co2 = fetch_openml(data_id=41187, as_frame=True)
co2.frame.head()

# %% [markdown]
#
# First, we create a date index and select only the CO2 column.

# %%
import pandas as pd

co2_data = co2.frame
co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
co2_data = co2_data[["date", "co2"]].set_index("date")
co2_data.head()

# %%
co2_data.index.min(), co2_data.index.max()

# %% [markdown]
#
# The data shows CO2 concentration measurements from March 1958 to December 2001.
# Let's plot this raw information.

# %%
_, ax = plt.subplots(figsize=(8, 6))
co2_data.plot(ax=ax)
ax.set(
    xlabel="Date",
    ylabel="CO$_2$ concentration (ppm)",
    title="Raw air samples measurements from\nthe Mauna Loa Observatory",
)
plt.show()

# %% [markdown]
# We take a monthly average and drop months without measurements. This smooths
# the data.

_, ax = plt.subplots(figsize=(8, 6))
co2_data = co2_data.resample("ME").mean().dropna(axis="index", how="any")
co2_data.plot(ax=ax)
ax.set(
    ylabel="Monthly average of CO$_2$ concentration (ppm)",
    title=(
        "Monthly average of air samples measurements\n from the Mauna Loa Observatory",
    ),
)
plt.show()

# %% [markdown]
#
# We want to predict the CO2 concentration based on the date. We also want to
# extrapolate values for years after 2001.
#
# First, we split the data and target. We convert the dates into numeric values.

# %%
X_train = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
y_train = co2_data["co2"].to_numpy()
y_mean = y_train.mean()
y_train -= y_mean

# %% [markdown]
#
# ### Exercise
#
# Design a kernel for a Gaussian Process that fits the training data. Then
# extrapolate values up to 2025.

# %%
import datetime
import numpy as np

today = datetime.datetime.now()
current_month = today.year + today.month / 12
X_test = np.linspace(start=1958, stop=current_month, num=1_000).reshape(-1, 1)

# %%
from sklearn.gaussian_process.kernels import (
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)


long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)
seasonal_kernel = (
    2.0**2
    * RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
)
irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
)
co2_kernel = (
    long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
)

# %%
gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False)
gaussian_process.fit(X_train, y_train)

# %%
mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)
mean_y_pred += y_mean

# %%
_, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    X_train, y_train + y_mean, color="black", linestyle="dashed", label="Measurements"
)
ax.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
ax.fill_between(
    X_test.ravel(),
    mean_y_pred - std_y_pred,
    mean_y_pred + std_y_pred,
    color="tab:blue",
    alpha=0.2,
)
ax.legend()
ax.set_xlabel("Year")
ax.set_ylabel("Monthly average of CO$_2$ concentration (ppm)")
_ = ax.set_title(
    "Monthly average of air samples measurements\n" "from the Mauna Loa Observatory"
)

# %%
