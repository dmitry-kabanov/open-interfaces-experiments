.PHONY : all
all : notebook

.PHONY : notebook
notebook :
	jupyter-book build oif-notebook

.PHONY : open-notebook
open-notebook :
	open oif-notebook/_build/html/index.html
