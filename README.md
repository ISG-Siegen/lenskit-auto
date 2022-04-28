# LensKit Project Template

This repository is a template for LensKit projects.  It is intended to be used
as the base for each repository, so that we can use `git merge` to incorporate
new changes to infrastructure files.

## Creating a Project

Here are the steps to create a project:

1.  Create a new GitHub repository for your project
2.  Clone your empty GitHub repository to your computer
3.  Add this repository as a remote:
   
        git remote add common https://github.com/lenskit/lk-common.git
        git fetch common

4.  Merge its `main` branch into your `main`:

        git merge common/main

5.  Update the files for your project.  See at least:

    - pyproject.toml
    - docs/conf.py
    - README.md

    There are FIXME comments in key files to indicate where they must be edited

6.  If your project does not need the MovieLens test data, remove that directory.

Then create your Python package directory and start working on code!

## Updating Common Files

To make sure you have the current version of the common files, run:

    git pull common main

You may need to resolve conflicts, many of these will be resolved in favor of your
project (the existing changes).  We recommend doing each common update as a pull
request to make sure tests and everything still pass.

## Common Design

Here are the common design parameters of LensKit projects:

- Packaging managed with [flit](https://flit.pypa.io/en/latest/).
- Docs live in `docs/`, build with Sphinx
- Tests live in `tests/`, run with PyTest
- Lightweight quality control with flake8
- MIT-licensed
- Test coverage measured with pytest-cov
