OIF_DIR := ./code

CC := /usr/bin/clang-14
CFLAGS := -march=native -mtune=native -O3 -g -fPIC
CFLAGS += -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt
CFLAGS += -ffunction-sections -pipe
CFLAGS += -fomit-frame-pointer -funroll-loops -fno-math-errno
CFLAGS += -fno-trapping-math
# CFLAGS += -fsanitize=address
CPPFLAGS := -I$(OIF_DIR)/oif/include -I$(OIF_DIR)/oif/interfaces/c/include/
CPPFLAGS += -I$(OIF_DIR)/oif_impl/lang_c/include/     -D_XOPEN_SOURCE

SHLIB_FLAGS := -shared -fPIC

LDFLAGS := -L$(OIF_DIR)/build
LDLIBS := -loif_c -loif_c_util -loif_c_data_structures -ldl -lm

# Local variables.
all_targets := $(addprefix bin/,burgers.so run_burgers_oif run_burgers_raw)
plot_script := python code/examples/plot_solution_1D.py


.PHONY : all
all : $(all_targets)


.PHONY : plot-oif
plot-oif:
	$(plot_script) _output/solution_dopri5c_oif.txt _assets/solution_dopri5c-oif.pdf


.PHONY : plot-raw
plot-raw:
	$(plot_script) _output/solution_dopri5c_raw.txt _assets/solution_dopri5c-raw.pdf

.PHONY : clean
clean :
	-$(RM) $(all_targets)

.PHONY : clobber
clobber : clean
	-$(RM) -r _output/*
	-$(RM) -r _assets/*


bin/burgers.so: burgers.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SHLIB_FLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

bin/run_burgers_oif: run_burgers_oif.c bin/burgers.so
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

bin/run_burgers_raw: run_burgers_raw.c bin/burgers.so
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
