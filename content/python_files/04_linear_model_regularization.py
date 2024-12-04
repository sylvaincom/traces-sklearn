# %% [markdown]
#
# # Regularization
#
# This notebook explores regularization in linear models.
#
# ## Introductory example
#
# We demonstrate a common issue with correlated features when fitting linear models.
#
# We use the penguins dataset to illustrate this issue.

# %%
# When using JupyterLite, uncomment and install the `skrub` package.
# %pip install skrub
import matplotlib.pyplot as plt
import skrub

skrub.patch_display()  # makes nice display for pandas tables

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins.csv")
penguins

# %% [markdown]
#
# We select features to predict penguin body mass. We remove rows with missing target values.

# %%
features = [
    "Island",
    "Clutch Completion",
    "Flipper Length (mm)",
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Species",
    "Sex",
]
target = "Body Mass (g)"
data, target = penguins[features], penguins[target]
target = target.dropna()
data = data.loc[target.index]
data

# %% [markdown]
#
# Let's evaluate a simple linear model using skrub's preprocessing.

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold

model = skrub.tabular_learner(estimator=LinearRegression())
model.set_output(transform="pandas")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    model, data, target, cv=cv, return_estimator=True, return_train_score=True
)
pd.DataFrame(cv_results)[["train_score", "test_score"]]

# %% [markdown]
#
# The test score looks good overall but performs poorly on one fold.
# Let's examine the coefficient values to understand why.

# %%
coefs = [est[-1].coef_ for est in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=cv_results["estimator"][0][-1].feature_names_in_)
coefs.plot.box(whis=[0, 100], vert=False)
plt.show()

# %% [markdown]
#
# **EXERCISE**
#
# What do you observe? What causes this behavior?
# Apply the preprocessing chain and check skrub's statistics on the resulting
# data to understand these coefficients.

# %%
# Write your code here.

# %% [markdown]
#
# ## Ridge regressor - L2 regularization
#
# We saw that coefficients can grow arbitrarily large when features correlate.
#
# $$
# loss = (y - X \beta)^2 + \alpha \|\beta\|_2
# $$
#
# L2 regularization forces weights toward zero. The parameter $\alpha$ controls
# this shrinkage. Scikit-learn implements this as the Ridge model. Let's fit it
# and examine its effect on weights.

# %%
from sklearn.linear_model import Ridge

model = skrub.tabular_learner(estimator=Ridge(alpha=1)).set_output(transform="pandas")

cv_results = cross_validate(
    model, data, target, cv=cv, return_estimator=True, return_train_score=True
)
pd.DataFrame(cv_results)[["train_score", "test_score"]]

# %%
coefs = [est[-1].coef_ for est in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=cv_results["estimator"][0][-1].feature_names_in_)
coefs.plot.box(whis=[0, 100], vert=False)
plt.show()

# %% [markdown]
#
# A small regularization solves the weight problem. We recover the original
# relationship:
#
# **EXERCISE**
#
# Try different $\alpha$ values and examine how they affect the weights.

# %%
# Write your code here.

# %% [markdown]
#
# ## Lasso regressor - L1 regularization
#
# L1 provides another regularization type. It follows this formula:
#
# $$
# loss = (y - X \beta)^2 + \alpha \|\beta\|_1
# $$
#
# Scikit-learn implements this as the Lasso regressor.

# %% [markdown]
#
# **EXERCISE**
#
# Repeat the previous experiment with different $\alpha$ values and examine how they
# affect the weights $\beta$.

# %%
# Write your code here.

# %% [markdown]
#
# ## Elastic net - Combining L2 and L1 regularization
#
# Combining L2 and L1 regularization offers unique benefits: it identifies important
# features while preventing non-zero coefficients from growing too large.

# %%
from sklearn.linear_model import ElasticNet

model = skrub.tabular_learner(estimator=ElasticNet(alpha=10, l1_ratio=0.95))
model.set_output(transform="pandas")

cv_results = cross_validate(
    model, data, target, cv=cv, return_estimator=True, return_train_score=True
)
pd.DataFrame(cv_results)[["train_score", "test_score"]]

# %%
coefs = [est[-1].coef_ for est in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=cv_results["estimator"][0][-1].feature_names_in_)
coefs.plot.box(whis=[0, 100], vert=False)
plt.show()

# %% [markdown]
#
# ## Hyperparameter tuning
#
# How do we choose the regularization parameter? The validation curve helps analyze
# single parameter effects. It plots scores versus parameter values.
#
# Let's use ValidationCurveDisplay to analyze how the alpha parameter affects
# Ridge regression.

# %%
model = skrub.tabular_learner(estimator=Ridge()).set_output(transform="pandas")

# %% [markdown]
#
# We need to find the parameter name for alpha in the model.

# %%
model.get_params()

# %%
import numpy as np
from sklearn.model_selection import ValidationCurveDisplay

disp = ValidationCurveDisplay.from_estimator(
    model,
    data,
    target,
    cv=cv,
    std_display_style="errorbar",
    param_name="ridge__alpha",
    param_range=np.logspace(-3, 3, num=20),
    n_jobs=2,
)
plt.show()

# %% [markdown]
#
# Too much regularization degrades model performance.
#
# **EXERCISE**
#
# Try a very small alpha (e.g. `1e-16`) and observe its effect on the
# validation curve.

# %%
# Write your code here.

# %% [markdown]
#
# In practice, we often use grid or random search instead of validation curves
# to choose regularization parameters. These methods run internal cross-validation
# to select the best-performing model. Let's demonstrate random search.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %%
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {"ridge__alpha": loguniform(1e-3, 1e3)}
search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=cv)
search.fit(data_train, target_train)

# %%
search.best_params_

# %%
pd.DataFrame(search.cv_results_)

# %% [markdown]
#
# This approach enables nested cross-validation. The inner loop selects parameters
# while the outer loop evaluates model performance.

# %%
cv_results = cross_validate(
    search, data, target, cv=cv, return_estimator=True, return_train_score=True
)
pd.DataFrame(cv_results)[["train_score", "test_score"]]

# %% [markdown]
#
# Some scikit-learn models efficiently search hyperparameters internally. Models with
# "CV" in their name, like RidgeCV, automatically find optimal regularization
# parameters.

# %%
from sklearn.linear_model import RidgeCV

model = skrub.tabular_learner(estimator=RidgeCV(alphas=np.logspace(-3, 3, num=100)))
model.set_output(transform="pandas")

cv_results = cross_validate(
    model, data, target, cv=cv, return_estimator=True, return_train_score=True
)
pd.DataFrame(cv_results)[["train_score", "test_score"]]

# %%
alphas = [est[-1].alpha_ for est in cv_results["estimator"]]
alphas

# %% [markdown]
#
# ## What about classification?
#
# Classification handles regularization differently. Instead of creating new estimators,
# regularization becomes a model parameter. LogisticRegression and LinearSVC offer
# two main models. Both use penalty and C parameters (C inverts regression's alpha).
#
# We'll explore parameter C with LogisticRegression. First, let's load classification
# data to predict penguin species from culmen measurements.

# %%
data = pd.read_csv("../datasets/penguins_classification.csv")
data = data[data["Species"].isin(["Adelie", "Chinstrap"])]
data["Species"] = data["Species"].astype("category")
data.head()

# %%
X, y = data[["Culmen Length (mm)", "Culmen Depth (mm)"]], data["Species"]

# %%
import matplotlib.pyplot as plt

data.plot.scatter(
    x="Culmen Length (mm)",
    y="Culmen Depth (mm)",
    c="Species",
    edgecolor="black",
    s=50,
)
plt.show()

# %% [markdown]
#
# **QUESTION**
#
# What regularization does LogisticRegression use by default? Check the documentation.
#
# Let's fit a model and visualize its decision boundary.

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)

# %%
from sklearn.inspection import DecisionBoundaryDisplay

display = DecisionBoundaryDisplay.from_estimator(
    model,
    X,
    response_method="decision_function",
    cmap=plt.cm.RdBu,
    plot_method="pcolormesh",
    shading="auto",
)
data.plot.scatter(
    x="Culmen Length (mm)",
    y="Culmen Depth (mm)",
    c="Species",
    edgecolor="black",
    s=50,
    ax=display.ax_,
)
plt.show()

# %%
coef = pd.Series(model.coef_[0], index=X.columns)
coef.plot.barh()
plt.show()

# %% [markdown]
#
# This example establishes a baseline for studying parameter C effects.
# The logistic regression loss function is:
#
# $$
# loss = \frac{1 - \rho}{2} w^T w + \rho \|w\|_1 + C \log ( \exp (y_i (X \beta)) + 1)
# $$

# %% [markdown]
#
# **EXERCISE**
#
# Fit models with different C values and examine how they affect coefficients
# and decision boundaries.

# %%
# Write your code here.

# %% [markdown]
#
# The loss formula shows C affects the data term (error between true and predicted
# targets). In regression, alpha affects the weights instead. This explains why C
# inversely relates to alpha.
