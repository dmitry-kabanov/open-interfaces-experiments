OIF_DIR := ./code

# CC := /usr/bin/clang
CFLAGS := -march=nocona -mtune=native -O3 -g -fPIC
CFLAGS += -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt
CFLAGS += -ffunction-sections -pipe
CFLAGS += -fomit-frame-pointer -funroll-loops -fno-math-errno
CFLAGS += -fno-trapping-math
# CFLAGS += -fsanitize=address
CPPFLAGS = -I$(OIF_DIR)/oif/include -I$(OIF_DIR)/oif/interfaces/c/include/

LDFLAGS := -L$(OIF_DIR)/build
LDLIBS := -loif_c_data_structures -loif_c -loif_c_util -ldl -lm

.PHONY : all
all : run_from_c

run_from_c : call_from_c
	./call_from_c

call_from_c : burgers.so

%.so: %.o
	$(LINK.c) -shared $^ -o $@ $(LDLIBS)

.PHONY : clean
clean :
	rm -rf burgers.so call_from_c
