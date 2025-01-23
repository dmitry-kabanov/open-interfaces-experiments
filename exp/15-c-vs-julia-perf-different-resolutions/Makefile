OIF_DIR := ./code

CC := /usr/bin/clang-14
CFLAGS := -march=native -mtune=native -O3 -g -fPIC
CFLAGS += -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt
CFLAGS += -ffunction-sections -pipe
CFLAGS += -fomit-frame-pointer -funroll-loops -fno-math-errno
CFLAGS += -fno-trapping-math
# CFLAGS += -fsanitize=address
CPPFLAGS = -I$(OIF_DIR)/oif/include -I$(OIF_DIR)/oif/interfaces/c/include/

LDFLAGS := -L$(OIF_DIR)/build
LDLIBS := -loif_c_data_structures -loif_c -loif_c_util -ldl -lm

.PHONY : all
all : bin/call_from_c

bin/call_from_c : burgers.so

%.so: %.o
	$(LINK.c) -shared $^ -o $@ $(LDLIBS)

bin/%: %.o | make-bindir
	echo $^
	$(LINK.c) $^ -o $@ $(LDLIBS)

make-bindir:
	mkdir -p bin

.PHONY : clean
clean :
	rm -rf burgers.so bin/call_from_c

# Delete all obtained results and compiled artifacts.
.PHONY : clobber
clobber : clean
	$(RM) -r _output/*
	$(RM) -r _assets/*
