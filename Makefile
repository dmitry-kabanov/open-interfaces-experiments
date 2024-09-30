.PHONY : all
all : notebook

.PHONY : notebook
notebook :
	jupyter-book build oif-notebook
