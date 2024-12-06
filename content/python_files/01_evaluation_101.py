# %% [markdown]
#
# # Evaluating a predictive model
#
# This notebook will:
#
# - Introduce linear models for regression tasks
# - Demonstrate scikit-learn's user API
# - Explain training and testing error concepts
# - Cover cross-validation techniques
# - Compare models against baselines
#
# ## Linear regression introduction
#
# Let's start with linear regression fundamentals. We'll use only NumPy initially,
# before introducing scikit-learn. First, we load our dataset.

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

# %% [markdown]
#
# Our dataset contains penguin flipper lengths and body masses. We want to predict a
# penguin's body mass from its flipper length. Since we predict a continuous value,
# this is a regression problem.
#
# Let's visualize the relationship between these measurements:

# %%
from matplotlib import pyplot as plt

ax = data.plot.scatter(x=data.columns[0], y=data.columns[1])
ax.set_title("Can I predict penguins' body mass?")
plt.show()

# %% [markdown]
#
# The data shows a clear linear trend - longer flippers correlate with heavier penguins.
# We'll model this relationship linearly.
#
# In this example:
#
# - Flipper length serves as our feature (predictor variable)
# - Body mass is our target (variable to predict)
#
# Each (flipper length, body mass) pair forms a sample. We train our model on these
# feature/target pairs. At prediction time, we use only features to predict potential
# targets. To evaluate our model, we compare its predictions against known targets.
#
# Throughout this notebook, we use:
# - `X`: feature matrix with shape `(n_samples, n_features)`
# - `y`: target vector with shape `(n_samples,)`

# %%
X, y = data[["Flipper Length (mm)"]], data[["Body Mass (g)"]]

# %% [markdown]
#
# We model the X-y relationship linearly as:
#
# $$
# y = X \beta
# $$
#
# where $\beta$ represents our model coefficients. For all features, this expands to:
#
# $$
# y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n
# $$
#
# Here we have only one feature coefficient $\beta_1$ for flipper length.
#
# Finding the optimal $\beta$ means finding values that minimize prediction error. We
# calculate $\beta$ using:
#
# $$
# X^T y = X^T X \beta
# $$
#
# which gives us:
#
# $$
# \beta = (X^T X)^{-1} X^T y
# $$

# %% [markdown]
#
# **EXERCISE**
#
# 1. Use NumPy to find $\beta$ ($\beta_0$ and $\beta_1$) using the normal equation
# 2. Calculate predictions using your $\beta$ values and X
# 3. Plot the original data (X vs y) and overlay your model's predictions

# %%
# Write your code here.

# %% [markdown]
#
# ## Scikit-learn API introduction
#
# Scikit-learn uses Python classes to maintain model state. These classes provide:
#
# - A `fit` method to learn parameters
# - A `predict` method to generate predictions
#
# **EXERCISE**
#
# Create a Python class that implements the linear model from above with:
#
# - A `fit` method to compute $\beta$
# - A `predict` method that outputs predictions for input X

# %%
# Write your code here.

# %% [markdown]
#
# Now let's use scikit-learn's built-in `LinearRegression` model instead of our
# implementation.

# %%
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

# %% [markdown]
#
# Scikit-learn models store fitted parameters as attributes ending in underscore.
# Our linear model stores `coef_` and `intercept_`:

# %%
model.coef_, model.intercept_

# %% [markdown]
#
# ## Cross-validation for proper model evaluation
#
# Let's evaluate our model's performance:

# %%
from sklearn.metrics import r2_score

score = r2_score(y, model.predict(X))
print(f"Model score: {score:.2f}")
# %% [markdown]
#
# This evaluation has a flaw. A model that simply memorizes training data would score
# perfectly. We need separate training and testing datasets to truly assess how well
# our model generalizes to new data. The training error measures model fit, while
# testing error measures generalization ability.
#
# Let's split our data into training and testing sets:

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model.fit(X_train, y_train)
model.coef_, model.intercept_

# %%
train_score = r2_score(y_train, model.predict(X_train))
print(f"Training score: {train_score:.2f}")

# %%
test_score = r2_score(y_test, model.predict(X_test))
print(f"Testing score: {test_score:.2f}")

# %% [markdown]
#
# Our model performs slightly worse on test data than training data. For comparison,
# let's examine a decision tree model which can show more dramatic differences:

# %%
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# %%
train_score = r2_score(y_train, model.predict(X_train))
print(f"Training score: {train_score:.2f}")

# %%
test_score = r2_score(y_test, model.predict(X_test))
print(f"Testing score: {test_score:.2f}")

# %% [markdown]
#
# Returning to our linear model: while we see small differences between training and
# testing scores, we can't determine if these differences are significant or just
# random variation from our data split. Cross-validation helps us estimate score
# distributions rather than single points.
#
# Cross-validation repeatedly evaluates the model using different train/test splits
# to account for variation in both fitting and prediction.

# %%
model = LinearRegression()

# %% [markdown]
#
# Scikit-learn's `cross_validate` function handles this repeated evaluation:

# %%
from sklearn.model_selection import KFold, cross_validate

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring="r2",
    return_train_score=True,
)

# %%
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %%
cv_results[["train_score", "test_score"]].mean()

# %%
cv_results[["train_score", "test_score"]].std()

# %% [markdown]
#
# Our results show similar train and test scores, though test scores vary more.
# Let's use repeated k-fold cross-validation to get more estimates and visualize
# the score distributions:

# %%
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_repeats=10, n_splits=3, random_state=42)
cv_results = cross_validate(model, X, y, cv=cv, scoring="r2", return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %%
ax = cv_results[["train_score", "test_score"]].plot.hist(alpha=0.7)
ax.set(xlim=(0, 1), title="Distribution of the scores with repeated k-fold")
plt.show()

# %% [markdown]
#
# The similar performance on training and testing sets with low variation indicates
# our model generalizes well.

# %% [markdown]
#
# **EXERCISE**
#
# Repeat the cross-validation using `KFold` with `shuffle=False`. Compare and explain
# the differences from our previous analysis.

# %%
# Write your code here.

# %% [markdown]
#
# ## Baseline model comparison
#
# It is common to compare the performance of a new model against simple models.
# These baseline models do not necessarily have to learn anything from the data.
# But they provide a reference to compare against.
#
# **EXERCISE**
#
# Compare your linear model against such a baseline:
#
# 1. Use cross-validation to get 30+ score estimates
# 2. Try a `DummyRegressor` that predicts the mean target value of the training set
# 3. Use `permutation_test_score` function to estimate the performance of a random model
# 4. Plot test score distributions for all three models

# %%
# Write your code here.

# %% [markdown]
#
# ## Model uncertainty evaluation

# Cross-validation evaluates uncertainty in the full fit/predict process by training
# different models on each cross-validation split.
#
# For a single model, we can evaluate prediction uncertainty through bootstrapping:

# %%
model = LinearRegression()
model.fit(X_train, y_train)

# %% [markdown]
#
# **EXERCISE**
#
# 1. Generate model predictions on the test set
# 2. Create 100 bootstrap prediction samples using `np.random.choice`
# 3. Plot the bootstrap sample distribution

# %%
# Write your code here.
