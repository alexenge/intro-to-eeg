# Allows you to type `make` or `make all` from the command line to build the
# book, first automatically converting all Python scripts (in the `py/` folder)
# to Jupyter notebooks (in the `ipynb/` folder) using `jupytext`.
all:
	mkdir ipynb
	jupytext --to ipynb py/*.py
	mv py/*.ipynb ipynb/
	jupyter-book build . --all
