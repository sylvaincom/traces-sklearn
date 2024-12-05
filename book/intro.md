# Introduction to machine learning

This tutorial introduces how to use `scikit-learn` to craft predictive models using
machine learning.

## Static version of the course

You can preview the course as a JupyterBook. The table of contents is available below.
If you want to execute the notebooks yourself, refer to the next section.

### Table of contents

```{tableofcontents}
```

## Executable version of the course

The following dependencies are required for the course:

- `jupyterlab`
- `jupytext`
- `notebook`
- `numpy`
- `scipy`
- `scikit-learn`
- `skrub`
- `pandas`
- `pyarrow`
- `matplotlib`
- `seaborn`

We offer several ways to run the course locally. Depending on your favorite package
manager, you can use one of the following options:

- JupyterLite: if you want to avoid installing anything on your computer.
- `pixi`: if you want the latest cutting-edge technology.
- `conda`: if you want to stick to a more traditional approach.
- `pip`: if you want to use the standard Python package manager.

### Use JupyterLite

JupyterLite is JupyterLab distribution running in the browser. It uses the Pyodide
kernel. In short, you can click on the badge below to start the course in your
browser. The lecture notes are located in `content/notebooks`.

[![Launch JupyterLite](/images/jupyterlite_badge.svg 'Our JupyterLite website')](https://glemaitre.github.io/traces-sklearn/jupyterlite)

Here, we describe the pros and cons of this approach.

**Pros**:

- No installation required
- Fast to start
- No need to configure Python environment

**Cons**:

- The execution of the first cell is always slow because it requires to potentially
  download the package and intialize the kernel.
- You will witness that we need to call `%pip install` to install a couple of packages
  in addition of the `import` statements in the notebook.
- We need to use `pyodide-http` to load some datasets when fetching from the internet.
- We need to make some defensive import when those are optional dependencies of
  some libraries, e.g. importing `matplotlib` when using `pandas` plot.

### Use `pixi`, `conda` or `pip`

#### Prerequisites

First clone the repository:

```bash
git clone https://github.com/glemaitre/traces-sklearn.git
```

Alternatively, download an archive at the
[following link](https://github.com/glemaitre/traces-sklearn/archive/refs/heads/main.zip).

#### Install the package manager

For `pixi`, refer to the [official website](https://pixi.sh/latest/#installation) for
installation.

For `conda`, download and install the latest version of `miniforge` from the [official
website](https://conda-forge.org/download/).

For `pip`, it is already installed if you have Python.

#### Install the dependencies

For `pixi`, you don't need to do anything. It will be automatically installed in the
next step.

For `conda`, you can install the dependencies using the `environment.yml` file:

```bash
conda env create --file environment.yml
```

For `pip`, you can install the dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### Launching Jupyter Lab

To launch Jupyter Lab, run the following command:

```bash
pixi run jupyter lab
```

The Python environment and necessary packages will be automatically installed for you.

For `conda`, you need to activate the environment:

```bash
conda activate traces-sklearn
```

Then, for `conda` and `pip`, you can launch Jupyter Lab with:

```bash
jupyter lab
```

#### Opening lecture notes

The lecture notes are available in the `python_files` directory. To open the Python
file as notebook, you need to right click on the file and select
`Open with` -> `Notebook`. This is using `jupytext` to interpret those files as
notebooks.

Alternatively, you convert those files into notebooks.

With `pixi`, you can run:

```bash
pixi run -e docs convert-to-notebooks
```

With `conda` and `pip`, you can run the `jupytext` command:

```bash
jupytext --to notebook ./content/python_files/*.py
mkdir -p ./content/notebooks
mv ./content/python_files/*.ipynb ./content/notebooks
```
