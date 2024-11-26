.PHONY : all
all : notebook

.PHONY : notebook
notebook :
	rm -rf oif-notebook/_build && jupyter-book build oif-notebook

.PHONY : open-notebook
open-notebook : notebook
	open oif-notebook/_build/html/index.html

.PHONY : check-dirty-code-worktrees
check-dirty-code-worktrees :
	bin/check_dirty_code_worktrees

.PHONY : start-web-server
start-web-server : notebook
	cd oif-notebook/_build/html && python -m http.server
