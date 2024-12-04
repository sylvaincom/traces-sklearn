# %% [markdown]
#
# ## Beyond linear separations
#
# This notebook shows how preprocessing makes linear models flexible enough to fit data
# with non-linear relationships between features and targets.

# %%
# When using JupyterLite, uncomment and install the `skrub` package.
# %pip install skrub
import matplotlib.pyplot as plt
import skrub

skrub.patch_display()  # makes nice display for pandas tables

# %% [markdown]
#
# ## Limitation of linear separation
#
# We create a complex classification toy dataset where a linear model will likely fail.
# Let's generate the dataset and plot it.

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

feature_names = ["Feature #0", "Feature #1"]
target_name = "class"

X, y = make_moons(n_samples=100, noise=0.13, random_state=42)

# Store both data and target in a dataframe to ease plotting
moons = pd.DataFrame(
    np.concatenate([X, y[:, np.newaxis]], axis=1), columns=feature_names + [target_name]
)
moons[target_name] = moons[target_name].astype("category")
X, y = moons[feature_names], moons[target_name]

# %%
moons.plot.scatter(
    x=feature_names[0],
    y=feature_names[1],
    c=y,
    s=50,
    cmap=plt.cm.RdBu,
)
plt.show()

# %% [markdown]
#
# Looking at the dataset, we see that a linear separation cannot effectively
# discriminate between the classes.
#
# **EXERCISE**
#
# 1. Fit a `LogisticRegression` model on the dataset.
# 2. Use `sklearn.inspection.DecisionBoundaryDisplay` to draw the decision
#    boundary of the model.

# %%
# Write your code here.

# %% [markdown]
#
# **EXERCISE**
#
# 1. Fit a `LogisticRegression` model on the dataset but add a
#    `sklearn.preprocessing.PolynomialFeatures` transformer.
# 2. Use `sklearn.inspection.DecisionBoundaryDisplay` to draw the decision boundary of
#    the model.

# %%
# Write your code here.

# %% [markdown]
#
# ## What about SVM?

# Support Vector Machines (SVM) offer another family of linear algorithms. SVMs use a
# different training approach than logistic regression. The model finds a hyperplane
# that maximizes the margin to the closest points.

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

model = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC())])
model.fit(X, y)

# %%
from sklearn.inspection import DecisionBoundaryDisplay

display = DecisionBoundaryDisplay.from_estimator(model, X, cmap=plt.cm.RdBu)
moons.plot.scatter(
    x=feature_names[0], y=feature_names[1], c=y, s=50, cmap=plt.cm.RdBu, ax=display.ax_
)
plt.show()

# %% [markdown]
#
# SVMs become non-linear through the "kernel trick". This projects data into a higher
# dimensional space without explicitly building the kernel, only computing dot products.
# The `SVC` class enables kernel use. We use a polynomial kernel to create something
# similar to the previous pipeline with `PolynomialFeatures`.

# %%
from sklearn.svm import SVC

model = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="poly", degree=3))])
model.fit(X, y)

# %%
display = DecisionBoundaryDisplay.from_estimator(model, X, cmap=plt.cm.RdBu)
moons.plot.scatter(
    x=feature_names[0], y=feature_names[1], c=y, s=50, cmap=plt.cm.RdBu, ax=display.ax_
)
plt.show()

# %% [markdown]
#
# We can also use other kernel types, like the Radial Basis Function (RBF).

# %%
from sklearn.svm import SVC

model = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
model.fit(X, y)

# %%
display = DecisionBoundaryDisplay.from_estimator(model, X, cmap=plt.cm.RdBu)
moons.plot.scatter(
    x=feature_names[0], y=feature_names[1], c=y, s=50, cmap=plt.cm.RdBu, ax=display.ax_
)
plt.show()
# %% [markdown]
#
# Note that SVMs do not scale well with large datasets. Sometimes it works better to
# approximate the kernel explicitly with a transformer like `Nystroem`.

# %%
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    [("nystroem", Nystroem()), ("logistic_regression", LogisticRegression())]
)
model.fit(X, y)

# %%
display = DecisionBoundaryDisplay.from_estimator(model, X, cmap=plt.cm.RdBu)
moons.plot.scatter(
    x=feature_names[0], y=feature_names[1], c=y, s=50, cmap=plt.cm.RdBu, ax=display.ax_
)
plt.show()

# %% [markdown]
#
# The decision boundary looks similar to an SVM with an RBF kernel. Let's demonstrate
# the scaling limitations of SVM classifiers.

# %%
data = pd.read_csv("../datasets/adult-census-numeric-all.csv")
data.head()

# %%
target_name = "class"
X = data.drop(columns=target_name)
y = data[target_name]

# %%
X.shape

# %% [markdown]
#
# The dataset contains almost 50,000 samples - quite large for an SVM model.

# %% [markdown]
#
# **EXERCISE**
#
# 1. Split the dataset into training and testing sets.
# 2. Create a model with an RBF kernel SVM. Time how long it takes to fit.
# 3. Repeat with a model using Nystroem kernel approximation and logistic regression.
# 4. Compare the test scores of both models.

# %%
# Write your code here.
