# Allows you to type `make` or `make all` from the command line to build the
# book, first automatically converting all Python scripts (in the `py/` folder)
# to Jupyter notebooks (in the `ipynb/` folder) using `jupytext`.
all:
	mkdir -p ipynb
	jupytext --to ipynb py/*.py
	mv py/*.ipynb ipynb/
ifneq (, $(shell command -v git 2>/dev/null))
	git checkout -- ipynb/*.ipynb
endif
