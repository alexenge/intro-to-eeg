# If you type `make` from the command line in the root directory, and have GNU
# Make installed (pre-installed on Linux, try Google for Windows or Mac), this
# will:
#
# * Convert all Python scripts in (`py`) to Jupyter notebooks (`ipynb`)
# * Build the Jupyter Book from the Jupyter notebooks
# * Clean up the Jupyter notebooks (only if you have `git` installed)
#

all:
	mkdir -p ipynb
	jupytext --to ipynb py/*.py
	mv py/*.ipynb ipynb/
	jupyter-book build .
ifneq (, $(shell command -v git 2>/dev/null))
	git checkout -- ipynb/*.ipynb
endif
