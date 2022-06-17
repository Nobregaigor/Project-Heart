# Project Heart

Documentation is on the way! Not ready, but it is starting to take shape. You can find it here: [Docs](https://nobregaigor.github.io/Project-Heart/)

_As I am finalizing core functions and adding visualization methods, I will be adding new documentation. Some changes in the code might occur, since this project is still under developement._

# How to install:

We currently do not have a released version of this project, however, you can clone our repository and install the source code locally:

1. Clone our repo using `git clone https://github.com/Nobregaigor/Project-Heart`
2. If you have a virtual enviroment (recommended), activate it.
3. Go to the repo directory `cd ./Project-Heart` and install dependencies with `pip install -r requirements.txt`
4. Install (locally) Project-Heart with `pip install .`

### How to update:

As we are still working on this library, it is a good pratice to check for a new version of the repo; we might correct or add new content. The functions at `notebooks` directory should be working, as we try to monitor them at every new update.

To check and update your local repository with the lastest version you can do:

1. Go to directory you cloned this repo with `cd path/to/project-heart`
2. Fetch new content with `git fetch`
3. Pull new content with `git pull`


## Additional dependencies:

- To read/write .feb and read .xplt:

  - Reference: [febio-python](https://github.com/Nobregaigor/febio-python)
  - Install: `pip install febio-python==0.1.4`

- For Tetrahedralizer backends:

  - Reference: [Wildmeshing](https://wildmeshing.github.io/python/)
  - Install: `conda install -c conda-forge wildmeshing`
  - Reference: [Tetgen](https://tetgen.pyvista.org/)
  - Install: `pip install tetgen`

- For LV fiber computation:
  - Reference: [LDRB](https://github.com/finsberg/ldrb/)
  - Install: `conda install -c conda-forge ldrb`

Note: we currently only support `python==3.9.7` due to dependencies.
