# Project Heart

--- DOCUMENTATION PENDING (still under development) ---


![image](https://user-images.githubusercontent.com/42748242/162364034-682412ea-9753-4ab5-8985-abded0a8e75b.png)


# How to install:

We currently do not have a released version of this project, however, you can clone our repository and install the source code locally:

1) Clone our repo using `git clone https://github.com/Nobregaigor/Project-Heart`
2) If you have a virtual enviroment (recommended), activate it.
3) Go to the repo directory `cd ./Project-Heart` and install dependencies with `pip install -r requirements.txt`
4) Install (locally) Project-Heart with `pip install .`

## Additional dependencies:

- For Tetrahedralizer backends:
  - Reference: [Wildmeshing](https://wildmeshing.github.io/python/)
  - Install: `conda install -c conda-forge wildmeshing`
  - Reference: [Tetgen](https://tetgen.pyvista.org/)
  - Install: `pip install tetgen`

- For LV fiber computation: 
  - Reference: [LDRB](https://github.com/finsberg/ldrb/)
  - Install: `conda install -c conda-forge ldrb`


Note: we currently only support `python==3.9.7` due to dependencies.
