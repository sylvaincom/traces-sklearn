# %% [markdown]
#
# # Gaussian process
#
# All models that we encounter up to know provide a point-estimate at the
# moment of prediction. However, none of the models (apart the
# `QuantileRegression` provide some confidence interval regarding the provided
# predictions.
#
# A family of model known as Gaussian Process allows to obtain such
# information. In this notebook, we will present the difference of this model
# compare to models that we already presented.
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
# In this first example, we will use the true generative process without
# adding any noise. For training the Gaussian Process regression, we will only
# select few samples.

# %%
rng = np.random.default_rng(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

# %% [markdown]
#
# The advantage of a Gaussian kernel is that we can craft a kernel by hand and compose
# together some base kernels. Here, we will use a radial basis function (RBF) kernel and
# a constant parameter to fit the amplitude.

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

# %% [markdown]
#
# In the previous method that we presented, we had a single model where we
# usually try to find the optimal parameters that best fit the dataset. In
# Gaussian Process, the paradigm is different: we deal with a distribution of
# models. We have some *apriori* defined by the prior distribution of the
# models. The training set will be combined with this prior to provide us a
# posterior distribution of models.
#
# First, let's have a look at the prior distribution of our Gaussian process.

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
ax.set(
    xlabel="$x$", ylabel="$f(x)$", title="Sample from the GP prior distribution"
)
plt.show()

# %% [markdown]
#
# The sample from the prior distribution are just random realisation initially. They are
# far from fitting our true generative model. However, from all the sample, we can
# indeed have a distribution of models. In this case, we can plot the mean and the 95%
# confidence interval.

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


plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(
    xlabel="$x$", ylabel="$f(x)$", title="GP prediction using only prior distribution"
)
plt.show()

# %% [markdown]
#
# Thus, if we plot the true generative process and the prediction, we are far to be
# happy about our modelisation.

# %%
plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
    color="tab:orange",
)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("$x$")
_ = plt.ylabel("$f(x)$")

# %% [markdown]
#
# Now, we fit a Gaussian process on these few training samples to see how they will
# influence the posterior distribution.

# %%
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

# %% [markdown]
#
# After fitting our model, we see that the hyperparameters of the kernel have been
# optimized. Now, we will use our kernel to compute the mean prediction of the full
# dataset and plot the 95% confidence interval.

# %%
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression \non noise-free dataset")

# %% [markdown]
#
# We see that for a prediction made on a data point close to the one from the training
# set, the 95% confidence has a small amplitude. Whenever a sample falls far from
# training data, our model's prediction is less accurate and the model prediction is
# less precise (higher uncertainty).
#
# ## Example with noisy targets
#
# We can repeat a similar experiment adding an additional noise to the target this time.
# It will allow seeing the effect of the noise on the fitted model.
#
# We add some random Gaussian noise to the target with an arbitrary standard deviation.

# %%
dy = 0.5 + 1.0 * rng.random_sample(y_train.shape)
y_train_noisy = y_train + rng.normal(0, dy)

# %% [markdown]
#
# We create a similar Gaussian process model. In addition to the kernel, this
# time, we specify the parameter `alpha` which can be interpreted as the
# variance of a Gaussian noise.

# %%
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=dy ** 2, n_restarts_optimizer=9
)
gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# %% [markdown]
#
# Let's plot the mean prediction and the uncertainty region as before.

# %%
plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.errorbar(
    X_train,
    y_train_noisy,
    dy,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression \non a noisy dataset")
