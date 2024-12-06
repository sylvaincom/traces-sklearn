# %% [markdown]
#
# # Quantile regression

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

# %%
X = penguins[["Flipper Length (mm)"]]
y = penguins["Body Mass (g)"]

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

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

model_estimate_mean = HistGradientBoostingRegressor(
    categorical_features="from_dtype", loss="squared_error"
)
model_estimate_mean.fit(X, y)

# %%
import numpy as np

X_test = pd.DataFrame(
    {"Flipper Length (mm)": np.linspace(X.min(axis=None), X.max(axis=None), 100)}
)
y_pred_mean = model_estimate_mean.predict(X_test)

# %%
import matplotlib.pyplot as plt

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

# %%
model_estimate_median = HistGradientBoostingRegressor(
    categorical_features="from_dtype", loss="quantile", quantile=0.5
)
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

# %%
model_estimate_10 = HistGradientBoostingRegressor(
    categorical_features="from_dtype", loss="quantile", quantile=0.1
)
model_estimate_90 = HistGradientBoostingRegressor(
    categorical_features="from_dtype", loss="quantile", quantile=0.9
)

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
