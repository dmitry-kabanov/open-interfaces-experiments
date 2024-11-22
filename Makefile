.PHONY : all
all : notebook

.PHONY : notebook
notebook :
	jupyter-book build oif-notebook

.PHONY : open-notebook
open-notebook :
	open oif-notebook/_build/html/index.html

.PHONY : check-dirty-code-worktrees
check-dirty-code-worktrees :
	bin/check_dirty_code_worktrees
