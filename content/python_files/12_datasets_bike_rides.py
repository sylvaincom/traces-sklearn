# %% [markdown]
#
# # The bike rides dataset
#
# This notebook presents the "Bike Ride" dataset. The dataset exists in the directory
# `datasets` in comma separated values (CSV) format.
#
# ## Presentation of the dataset
#
# We open this dataset using pandas.

# %%
# When using JupyterLite, uncomment and install the `skrub` package.
# %pip install skrub
import matplotlib.pyplot as plt
import skrub

skrub.patch_display()  # makes nice display for pandas tables

# %%
import pandas as pd

cycling = pd.read_csv("../datasets/bike_rides.csv")
cycling

# %% [markdown]
#
# The first column `timestamp` contains specific information about the time and date
# of each record while other columns contain numerical measurements. Let's check the
# data types of the columns in detail.

# %%
cycling.info()

# %% [markdown]
#
# CSV format stores data as text. Pandas infers numerical types by default. This
# explains why all features except `timestamp` appear as floating point values.
# However, the `timestamp` appears as an `object` column. This means the data in
# this column exists as `str` rather than a specialized `datetime` data type.
#
# We need to set an option to tell pandas to infer this data type when opening
# the file. Additionally, we want to use `timestamp` as an index. We reopen the
# file with extra arguments to help pandas read our CSV file properly.

# %%
cycling = pd.read_csv(
    "../datasets/bike_rides.csv", index_col=0, parse_dates=True
)
cycling.index.name = ""
cycling

# %%
cycling.info()

# %% [markdown]
#
# By telling pandas to parse the date, we get a `DatetimeIndex` that helps filter
# data based on date.
#
# Let's examine the data stored in our dataframe. This helps us frame the data
# science problem we aim to solve.
#
# The records include information from GPS recordings of a cyclist (`speed`,
# `acceleration`, `slope`) and extra information from other sensors: `heart-rate`
# shows the number of heart beats per minute, `cadence` indicates how fast the
# cyclist turns the pedals, and `power` measures the work required to move forward.
#
# To explain power more intuitively:
#
# Consider a soup blender used to blend vegetables. The blender's engine develops
# ~300 Watts of instantaneous power. Here, our cyclist acts as the engine (though
# an average cyclist develops ~150 Watts) and moving forward replaces blending
# vegetables.
#
# Professional cyclists use power to calibrate their training and track energy
# expenditure during rides. Higher power requires more energy, which demands more
# resources to create this energy. Humans use food as their resource. A soup
# blender might use uranium, petrol, natural gas, or coal. Our body works as a
# power plant to transform resources into energy.
#
# The challenge with measuring power relates to sensor cost: a cycling power meter
# costs between $400 and $1000. This leads to our data science problem: predict
# instantaneous cyclist power using other (cheaper) sensors.

# %%
target_name = "power"
data, target = cycling.drop(columns=target_name), cycling[target_name]

# %% [markdown]
#
# Let's examine the target distribution first.

# %%
_, ax = plt.subplots()
target.plot.hist(bins=50, edgecolor="black", ax=ax)
ax.set_xlabel("Power (W)")
plt.show()

# %% [markdown]
#
# We see a peak at 0 Watts, representing moments when the cyclist stops pedaling
# (during descents or stops). On average, this cyclist delivers ~200 Watts. A long
# tail extends from ~300 Watts to ~400 Watts. This range represents efforts a
# cyclist trains to reproduce for breakaways in the final kilometers of a race.
# However, the human body finds it costly to maintain this power output.
#
# Let's examine the data.

# %%
data

# %% [markdown]
#
# First, let's look closely at the dataframe index.

# %%
data.index

# %% [markdown]
#
# The records occur every second.

# %%
data.index.min(), data.index.max()

# %% [markdown]
#
# The data spans from August 18, 2020 to September 13, 2020. Obviously, our cyclist
# did not ride every second between these dates. Only a few dates should appear in
# the dataframe, matching the number of cycling rides.

# %%
data.index.normalize().nunique()

# %% [markdown]
#
# Four different dates correspond to four rides. Let's extract only the first ride
# from August 18, 2020.

# %%
date_first_ride = "2020-08-18"
cycling_ride = cycling.loc[date_first_ride]
data_ride, target_ride = data.loc[date_first_ride], target.loc[date_first_ride]

# %%
_, ax = plt.subplots()
data_ride.plot(ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.set_title("Sensor values for different cyclist measurements")
plt.show()

# %% [markdown]
#
# Different units and ranges for each measurement (feature) make the plot hard to
# interpret. Also, high temporal resolution obscures observations. Let's resample
# the data for a smoother visualization.

# %%
data_ride.resample("60S").mean().plot()
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.set_title("Sensor values for different cyclist measurements")
plt.show()

# %% [markdown]
#
# Let's check the range of different features:

# %%
axs = data_ride.hist(figsize=(10, 12), bins=50, edgecolor="black", grid=False)
# add the units to the plots
units = [
    "beats per minute",
    "rotations per minute",
    "meters per second",
    "meters per second squared",
    "%",
]
for unit, ax in zip(units, axs.ravel()):
    ax.set_xlabel(unit)
plt.subplots_adjust(hspace=0.6)
plt.show()

# %% [markdown]
#
# These plots reveal interesting information: a cyclist spends time without
# pedaling. These samples correspond to null power. The slope also shows large
# extremes.
#
# Let's create a pair plot on a subset of data samples to confirm these insights.

# %%
import numpy as np

rng = np.random.default_rng(0)
indices = rng.choice(np.arange(cycling_ride.shape[0]), size=500, replace=False)

# %%
subset = cycling_ride.iloc[indices].copy()
# Quantize the target and keep the midpoint for each interval
subset["power"] = pd.qcut(subset["power"], 6, retbins=False)
subset["power"] = subset["power"].apply(lambda x: x.mid)

# %%
# install seaborn when you are using JupyterLite
# %pip install seaborn
import seaborn as sns

sns.pairplot(data=subset, hue="power", palette="viridis")
plt.show()

# %% [markdown]
#
# Low cadence correlates with low power. Higher slopes and heart-rates link to
# higher power: a cyclist needs more energy to climb hills, which demands more
# from the body. The interaction between slope and speed confirms this: lower
# speed with higher slope typically means higher power.
#
# ## Data science challenge
#
# This challenge asks you to predict cyclist power from other sensor measurements.
#
# Go beyond the baseline model! Here are some ideas:
#
# - Use physical models of the bike and cyclist to predict power (e.g. use
#   velocity and slope to predict power needed for climbing).
# - Try a black-box approach.
# - Predict confidence intervals.
# - Analyze sensor influence (i.e. feature importance).
# - Evaluate your models using cross-validation.
# - And more!
