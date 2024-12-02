# Introduction to scikit-learn for the TRACES program

This tutorial introduces how to use scikit-learn to craft predictive models using
machine learning.

## Browse online:

[![Launch JupyterBook](./book/images/jupyterbook_badge.svg 'Our JupyterBook
website')](https://glemaitre.github.io/traces-sklearn) [![Launch
JupyterLite](./book/images/jupyterlite_badge.svg 'Our JupyterLite
website')](https://glemaitre.github.io/traces-sklearn/jupyterlite)

## Getting started

### Install `pixi`

You can refer to the [official website](https://pixi.sh/latest/#installation) for
installation.

### Launching Jupyter Lab

To launch Jupyter Lab, run the following command:

```bash
pixi run jupyter lab
```

The Python environment and necessary packages will be automatically installed for you.

### Opening lecture notes

The lecture notes are available in the `content/python_files` directory. To open the
Python file as notebook, you need to right click on the file and select `Open with` ->
`Notebook`.

Alternatively, you can generate notebooks as well:

```bash
pixi run -e doc convert-to-notebooks
```

This will convert the Python files into notebooks in the folder `content/notebooks`.
