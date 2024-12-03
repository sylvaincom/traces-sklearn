# %% [markdown]
#
# # Data preprocessing
#
# This notebook explores preprocessing requirements for linear models.
#
# ## Numerical features
#
# Linear models are sensitive to data scale. While we did not preprocess data in the
# previous notebook, we should understand this sensitivity.
#
# Let's examine a simple example.

# %%
# When using JupyterLite, you will need to uncomment and install the `skrub` package.
# %pip install skrub
import matplotlib.pyplot as plt
import skrub

skrub.patch_display()  # make nice display for pandas tables

# %%
from sklearn.datasets import load_iris

data, target = load_iris(return_X_y=True, as_frame=True)
data

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(data, target)

# %% [markdown]
#
# The model raises a `ConvergenceWarning`. This indicates that it did not find weights
# that minimize the loss function.
#
# **EXERCISE**
#
# 1. `LogisticRegression` uses an LBFGS solver that iterates to find a solution. Check
#    how many iterations it took and compare with the default in the documentation.
# 2. Increase the number of iterations. Find the minimum number needed to avoid the
#    convergence warning.
# 3. Instead of increasing iterations, scale the data with `StandardScaler` before
#    fitting. Note the new iteration count.

# %%
# Write your code here.

# %% [markdown]
#
# We did not split the data into training and testing sets in the previous exercise.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %% [markdown]
#
# Scikit-learn's `Pipeline` is a powerful tool that chains transformations and a final
# estimator. We can connect a `StandardScaler` and `LogisticRegression` like this:

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline(
    steps=[("scaler", StandardScaler()), ("logistic_regression", LogisticRegression())]
)
model.fit(data_train, target_train)

# %%
print(f"Feature mean on training set: {model[0].mean_}")
print(f"Feature standard deviation on training set: {model[0].scale_}")

# %%
print(f"Number of iterations: {model[-1].n_iter_}")

# %% [markdown]
#
# **EXERCISE**
#
# The output shows that `StandardScaler` computed feature means and standard deviations
# from the training set. It uses these statistics to center and scale data before
# passing it to `LogisticRegression`.
#
# How do you think `StandardScaler` behaves on the test set? Consider this code:

# %%
from sklearn.metrics import accuracy_score

predicted_target = model.predict(data_test)

print(f"Accuracy on testing set: {accuracy_score(target_test, predicted_target):.3f}")

# %% [markdown]
#
# ## Categorical features
#
# We've shown how linear models benefit from feature scaling. Now let's examine
# categorical features using the penguins dataset.

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins.csv")
penguins

# %% [markdown]
#
# Categorical features take discrete values. Here's an example from the penguins dataset:

# %%
penguins["Sex"]

# %%
penguins["Sex"].value_counts()

# %% [markdown]
#
# These categories use non-numeric values. Models cannot process them directly, so we
# must convert categories to numbers.
#
# We can use two main strategies:
#
# - **Ordinal encoding**: Assigns a numeric value to each category
# - **One-hot encoding**: Creates binary features for each category

# %%
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
encoder.fit_transform(penguins[["Sex"]])

# %%
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder().set_output(transform="pandas")
encoder.fit_transform(penguins[["Sex"]])

# %% [markdown]
#
# **EXERCISE**
#
# 1. List advantages and disadvantages of both encoding strategies
# 2. Create a `Pipeline` that chains an encoder with a `LogisticRegression` model
# 3. Use cross-validation to evaluate model performance

# %%
# Write your code here.

# %% [markdown]
#
# ## Combine numerical and categorical features
#
# Scikit-learn's `ColumnTransformer` helps us handle both numerical and categorical
# features. Let's prepare our dataset:

# %%
categorical_features = ["Island", "Sex"]
numerical_features = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_name = "Species"

# %%
data = penguins[categorical_features + numerical_features]
target = penguins[target_name]
data

# %% [markdown]
#
# Our data contains missing values. For now, we'll simply drop rows with missing values
# in both data and target. We'll address this topic more thoroughly in the next section.

# %%
data = data.dropna()
target = target.loc[data.index]

# %%
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %%
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(), categorical_features),
    ]
)
preprocessor

# %% [markdown]
#
# The `ColumnTransformer` splits columns and sends each subset to its appropriate
# transformer.
#
# We can chain it with `LogisticRegression`:

# %%
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("logistic_regression", LogisticRegression()),
    ]
)
model

# %%
model.fit(data_train, target_train)

# %%
predicted_target = model.predict(data_test)
print(f"Accuracy: {accuracy_score(target_test, predicted_target):.3f}")

# %% [markdown]
#
# ## Dealing with missing values
#
# Let's reload our dataset with missing values intact:

# %%
categorical_features = ["Island", "Sex"]
numerical_features = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_name = "Species"

# %%
data = penguins[categorical_features + numerical_features]
target = penguins[target_name]
data

# %%
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %% [markdown]
#
# Try fitting the previous model again. What happens?

# %%
# Write your code here.

# %% [markdown]
#
# Models that don't handle missing values need imputation - replacing missing values
# with computed values from the data.
#
# **EXERCISE**
#
# Build a model that chains `ColumnTransformer`, `SimpleImputer`, and
# `LogisticRegression`.

# %%
# Write your code here.

# %% [markdown]
#
# ## `skrub` to help you out
#
# The `skrub` library offers utilities for baseline preprocessing. Use `tabular_learner`
# to quickly build a pipeline:

# %%
model = skrub.tabular_learner(estimator=LogisticRegression())
model

# %%
model.fit(data_train, target_train)

# %%
predicted_target = model.predict(data_test)
print(f"Accuracy: {accuracy_score(target_test, predicted_target):.3f}")
