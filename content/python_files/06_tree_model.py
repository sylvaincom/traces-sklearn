# %% [markdown]
#
# # Decision trees
#
# We explore a class of algorithms based on decision trees. Decision trees are
# extremely intuitive at their core. They encode a series of "if" and "else" choices,
# similar to how a person makes a decision. The data determines which questions to ask
# and how to proceed for each answer.
#
# For example, to create a guide for identifying an animal found in nature,
# you might ask the following series of questions:
#
# - Is the animal bigger or smaller than a meter long?
#     + *bigger*: does the animal have horns?
#         - *yes*: are the horns longer than ten centimeters?
#         - *no*: does the animal wear a collar?
#     + *smaller*: does the animal have two or four legs?
#         - *two*: does the animal have wings?
#         - *four*: does the animal have a bushy tail?
#
# And so on. This binary splitting of questions forms the essence of a decision tree.

# Tree-based models offer several key benefits. First, they require little preprocessing
# of the data. They work with variables of different types (continuous and discrete)
# and remain invariant to feature scaling.
#
# Tree-based models are also "nonparametric", meaning they do not have a fixed set of
# parameters to learn. Instead, a tree model becomes more flexible with more data. In
# other words, the number of free parameters grows with the number of samples and is not
# fixed like in linear models.
#
# ## Decision tree for classification

# %%
# When using JupyterLite, uncomment and install the `skrub` package.
# %pip install skrub
import matplotlib.pyplot as plt
import skrub

skrub.patch_display()  # makes nice display for pandas tables

# %%
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=[[0, 0], [1, 1]], random_state=42)

# %%
import numpy as np
import pandas as pd

X = pd.DataFrame(X, columns=["Feature #0", "Feature #1"])
class_names = np.array(["class #0", "class #1"])
y = pd.Series(class_names[y], name="Classes").astype("category")
data = pd.concat([X, y], axis=1)

# %%
data.plot.scatter(
    x="Feature #0",
    y="Feature #1",
    c="Classes",
    s=50,
    edgecolor="black",
)
plt.show()

# %% [markdown]
#
# We create a function to create this scatter plot by passing 2 variables: `data`
# and `labels`.
#
# ### Train a decision tree classifier
#
# We learn a set of binary rules using a portion of the data. Using these rules,
# we predict on the testing data.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, X_train, X_test, y_train, y_test = train_test_split(
    data, X, y, random_state=42
)

# %%
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)
pred

# %% [markdown]
#
# We plot the decision boundaries found using the training data.

# %%
from sklearn.inspection import DecisionBoundaryDisplay

display = DecisionBoundaryDisplay.from_estimator(tree, X_train, alpha=0.7)
data_train.plot.scatter(
    x="Feature #0", y="Feature #1", c="Classes", s=50, edgecolor="black", ax=display.ax_
)
plt.show()

# %% [markdown]
#
# Similarly, we get the following classification on the testing set.

display = DecisionBoundaryDisplay.from_estimator(tree, X_test, alpha=0.7)
data_test.plot.scatter(
    x="Feature #0", y="Feature #1", c="Classes", s=50, edgecolor="black", ax=display.ax_
)
plt.show()

# %% [markdown]
#
# We can also plot the tree structure.

# %%
from sklearn.tree import plot_tree

plot_tree(tree, feature_names=X.columns, class_names=class_names, filled=True)
plt.show()

# %% [markdown]
#
# **EXERCISE**
#
# 1. Modify the depth of the tree and observe how the partitioning evolves.
# 2. What can you conclude about under- and over-fitting of the tree model?
# 3. How would you choose the best depth?

# %% [markdown]
#
# Many parameters control the complexity of a tree, but maximum depth is perhaps the
# easiest to understand. This parameter limits how finely the tree can partition the
# input space, or how many "if-else" questions it can ask before deciding which class
# a sample belongs to.
#
# This parameter is crucial to tune for trees and tree-based models. The interactive
# plot below shows how underfitting and overfitting look for this model. A
# ``max_depth`` of 1 clearly underfits the model, while a depth of 7 or 8 clearly
# overfits. The maximum possible depth for this dataset is 8, at which point each leaf
# contains samples from only a single class. We call these leaves "pure."
#
# In the interactive plot below, blue and red colors indicate the predicted class for
# each region. The shade of color indicates the predicted probability for that class
# (darker = higher probability), while yellow regions indicate equal predicted
# probability for either class.
#
# ### Note about partitioning in decision trees
#
# In this section, we examine in more detail how a tree selects the best partition.
# First, we use a real dataset instead of synthetic data.

# %%
dataset = pd.read_csv("../datasets/penguins.csv")
dataset = dataset.dropna(subset=["Body Mass (g)"])
dataset.head()

# %% [markdown]
#
# We build a decision tree to classify penguin species using their body mass
# as a feature. To simplify the problem, we focus only on the Adelie and Gentoo species.

# %%
# Only select the column of interest
dataset = dataset[["Body Mass (g)", "Species"]]
# Make the species name more readable
dataset["Species"] = dataset["Species"].apply(lambda x: x.split()[0])
# Only select the Adelie and Gentoo penguins
dataset = dataset.set_index("Species").loc[["Adelie", "Gentoo"], :]
# Sort all penguins by their body mass
dataset = dataset.sort_values(by="Body Mass (g)")
# Convert the dataframe (2D) to a series (1D)
dataset = dataset.squeeze()
dataset

# %% [markdown]
#
# First, we examine the body mass distribution for each species.

# %%
_, ax = plt.subplots()
dataset.groupby("Species").plot.hist(ax=ax, alpha=0.7, legend=True)
ax.set_ylabel("Frequency")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %% [markdown]
#
# Instead of looking at the distribution, we can look at all samples directly.

# %%
# %pip install seaborn
import seaborn as sns

ax = sns.swarmplot(x=dataset.values, y=[""] * len(dataset), hue=dataset.index)
ax.set_xlabel(dataset.name)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %% [markdown]
#
# When we build a tree, we want to find splits that partition the data into groups that
# are as "unmixed" as possible. Let's make a first completely random split to highlight
# the principle.

# %%
# create a random state so we all get the same results
rng = np.random.default_rng(42)
random_idx = rng.choice(dataset.size)

ax = sns.swarmplot(x=dataset.values, y=[""] * len(dataset), hue=dataset.index)
ax.set_xlabel(dataset.name)
ax.set_title(f"Body mass threshold: {dataset[random_idx]} grams")
ax.vlines(dataset[random_idx], -1, 1, color="red", linestyle="--")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %% [markdown]
#
# After the split, we want two partitions where samples come from a single class as
# much as possible and contain as many samples as possible. Decision trees use a
# **criterion** to assess split quality. **Entropy** describes the class mixture in a
# partition. Let's compute the entropy for the full dataset and the sets on each side
# of the split.

# %%
from scipy.stats import entropy

dataset.index.value_counts()

parent_entropy = entropy(dataset.index.value_counts(normalize=True))
parent_entropy

left_entropy = entropy(dataset[:random_idx].index.value_counts(normalize=True))
left_entropy

right_entropy = entropy(dataset[random_idx:].index.value_counts(normalize=True))
right_entropy

# %% [markdown]
#
# We assess split quality by combining the entropies. This is called the
# **information gain**.

# %%
parent_entropy - (left_entropy + right_entropy)

# %% [markdown]
# However, we should normalize the entropies by the number of samples in each set.


# %%
def information_gain(labels_parent, labels_left, labels_right):
    # compute the entropies
    entropy_parent = entropy(labels_parent.value_counts(normalize=True))
    entropy_left = entropy(labels_left.value_counts(normalize=True))
    entropy_right = entropy(labels_right.value_counts(normalize=True))

    n_samples_parent = labels_parent.size
    n_samples_left = labels_left.size
    n_samples_right = labels_right.size

    # normalize with the number of samples
    normalized_entropy_left = (n_samples_left / n_samples_parent) * entropy_left
    normalized_entropy_right = (n_samples_right / n_samples_parent) * entropy_right

    return entropy_parent - normalized_entropy_left - normalized_entropy_right


information_gain(dataset.index, dataset[:random_idx].index, dataset[random_idx:].index)

# %% [markdown]
#
# Now we compute the information gain for all possible body mass thresholds.

# %%
all_information_gain = pd.Series(
    [
        information_gain(dataset.index, dataset[:idx].index, dataset[idx:].index)
        for idx in range(dataset.size)
    ],
    index=dataset,
)

# %%
ax = all_information_gain.plot()
ax.set_ylabel("Information gain")
plt.show()

# %%
ax = (all_information_gain * -1).plot(color="red", label="Information gain")
ax = sns.swarmplot(x=dataset.values, y=[""] * len(dataset), hue=dataset.index)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %% [markdown]
#
# The maximum information gain corresponds to the split that best partitions our data.
# Let's check the corresponding body mass threshold.

# %%
all_information_gain.idxmax()

ax = (all_information_gain * -1).plot(color="red", label="Information gain")
ax = sns.swarmplot(x=dataset.values, y=[""] * len(dataset), hue=dataset.index)
ax.vlines(all_information_gain.idxmax(), -1, 1, color="red", linestyle="--")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.show()

# %% [markdown]
#
# ## Decision Tree Regression

# %%
rnd = np.random.default_rng(42)
x = np.linspace(-3, 3, 100)
y_no_noise = np.sin(4 * x) + x
y = y_no_noise + rnd.normal(size=len(x))
X = x.reshape(-1, 1)

# %%
_, ax = plt.subplots()
ax.scatter(X, y, s=50)
ax.set(xlabel="Feature X", ylabel="Target y")
plt.show()

# %%
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(max_depth=2)
reg.fit(X, y)

# %%
X_test = np.linspace(-3, 3, 1000).reshape((-1, 1))
y_test = reg.predict(X_test)

_, ax = plt.subplots()
ax.plot(X_test.ravel(), y_test, color="tab:blue", label="prediction")
ax.plot(X.ravel(), y, "C7.", label="training data")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %% [markdown]
#
# A single decision tree estimates the signal non-parametrically but has some issues.
# In some regions, the model shows high bias and underfits the data
# (seen in long flat lines that don't follow data contours),
# while in other regions it shows high variance and overfits
# (seen in narrow spikes influenced by noise in single points).
#
# **EXERCISE**
#
# 1. Take the above example and repeat the training/testing by changing the tree depth.
# 2. What can you conclude?

# %% [markdown]
#
# ## Other tree hyperparameters
#
# The max_depth hyperparameter controls overall tree complexity. This parameter works
# well when a tree is symmetric. However, trees are not guaranteed to be symmetric. In
# fact, optimal generalization might require some branches to grow deeper than others.
#
# We build a dataset to illustrate this asymmetry. We generate a dataset with 2 subsets:
# one where the tree should find a clear separation and another where samples from both
# classes mix. This means a decision tree needs more splits to properly classify samples
# from the second subset than from the first subset.

# %%
from sklearn.datasets import make_blobs

feature_names = ["Feature #0", "Feature #1"]
target_name = "Class"

# Blobs that will be interlaced
X_1, y_1 = make_blobs(n_samples=300, centers=[[0, 0], [-1, -1]], random_state=42)
# Blobs that will be easily separated
X_2, y_2 = make_blobs(n_samples=300, centers=[[3, 6], [7, 0]], random_state=42)

X = np.concatenate([X_1, X_2], axis=0)
y = np.concatenate([y_1, y_2])
data = np.concatenate([X, y[:, np.newaxis]], axis=1)
data = pd.DataFrame(data, columns=feature_names + [target_name])
data[target_name] = data[target_name].astype(np.int64).astype("category")
data

# %%
_, ax = plt.subplots(figsize=(10, 8))
data.plot.scatter(
    x="Feature #0",
    y="Feature #1",
    c="Class",
    s=100,
    edgecolor="black",
    ax=ax,
)
ax.set_title("Synthetic dataset")
plt.show()

# %% [markdown]
#
# First, we train a shallow decision tree with max_depth=2. This depth should suffice
# to separate the easily separable blobs.

# %%
max_depth = 2
tree = DecisionTreeClassifier(max_depth=max_depth)
tree.fit(X, y)

_, ax = plt.subplots(figsize=(10, 8))
DecisionBoundaryDisplay.from_estimator(tree, X, cmap=plt.cm.RdBu, ax=ax)
data.plot.scatter(
    x="Feature #0",
    y="Feature #1",
    c="Class",
    s=100,
    edgecolor="black",
    ax=ax,
)
ax.set_title(f"Decision tree with max-depth of {max_depth}")
plt.show()

# %% [markdown]
#
# As expected, the blue blob on the right and red blob on top separate easily. However,
# we need more splits to better separate the mixed blobs.
#
# The red blob on top and blue blob on the right separate perfectly. However, the tree
# still makes mistakes where the blobs mix. Let's examine the tree structure.

# %%
_, ax = plt.subplots(figsize=(15, 8))
plot_tree(
    tree, feature_names=feature_names, class_names=class_names, filled=True, ax=ax
)
plt.show()

# %% [markdown]
#
# The right branch achieves perfect classification. Now we increase the depth to see
# how the tree grows.

# %%
max_depth = 6
tree = DecisionTreeClassifier(max_depth=max_depth)
tree.fit(X, y)

# %%
_, ax = plt.subplots(figsize=(10, 8))
DecisionBoundaryDisplay.from_estimator(tree, X, cmap=plt.cm.RdBu, ax=ax)
data.plot.scatter(
    x="Feature #0",
    y="Feature #1",
    c="Class",
    s=100,
    edgecolor="black",
    ax=ax,
)
ax.set_title(f"Decision tree with max-depth of {max_depth}")
plt.show()

# %%
_, ax = plt.subplots(figsize=(25, 15))
plot_tree(
    tree, feature_names=feature_names, class_names=class_names, filled=True, ax=ax
)
plt.show()
# %% [markdown]
#
# As expected, the left branch continues to grow while the right branch stops splitting.
# Setting max_depth cuts the tree horizontally at a specific level, whether or not a
# branch would benefit from growing further.
#
# The hyperparameters min_samples_leaf, min_samples_split, max_leaf_nodes, and
# min_impurity_decrease allow asymmetric trees and apply constraints at the leaf or
# node level. Let's examine the effect of min_samples_leaf.

# %%
min_samples_leaf = 20
tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
tree.fit(X, y)

_, ax = plt.subplots(figsize=(10, 8))
DecisionBoundaryDisplay.from_estimator(tree, X, cmap=plt.cm.RdBu, ax=ax)
data.plot.scatter(
    x="Feature #0",
    y="Feature #1",
    c="Class",
    s=100,
    edgecolor="black",
    ax=ax,
)
ax.set_title(f"Decision tree with leaf having at least {min_samples_leaf} samples")
plt.show()

# %%
_, ax = plt.subplots(figsize=(15, 15))
plot_tree(
    tree, feature_names=feature_names, class_names=class_names, filled=True, ax=ax
)
plt.show()
# %% [markdown]
#
# This hyperparameter ensures leaves contain a minimum number of samples and prevents
# further splits otherwise. These hyperparameters offer an alternative to the max_depth
# parameter.
