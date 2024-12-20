BUILD_TYPE ?= release  # Values: [ debug, release ]

CC := /usr/bin/clang
CPPFLAGS := \
    -I$(OIF_DIR)/oif/include \
    -I$(OIF_DIR)/oif/interfaces/c/include/ \
    -I$(OIF_DIR)/oif_impl/lang_c/include/ \
    -D_XOPEN_SOURCE

CFLAGS := \
    -std=c17 \
    -Wall -Wextra -Wpedantic -Werror \
    -O3 \
    -g \
    -march=native \
    -mtune=native -fomit-frame-pointer -funroll-loops -fno-math-errno -fno-trapping-math

ifeq ($(BUILD_TYPE),debug)
CFLAGS := \
    -O0 \
    -g
endif

SHLIB_FLAGS := -shared -fPIC

LDFLAGS := -L$(OIF_DIR)/build
LDLIBS := -loif_c -loif_c_util -loif_c_data_structures -ldl -lm

# Local variables.
all_targets := burgers.so run_burgers_oif run_burgers_raw
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
clean:
	-$(RM) $(all_targets)


burgers.so: burgers.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SHLIB_FLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

run_burgers_oif: run_burgers_oif.c burgers.so
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

run_burgers_raw: run_burgers_raw.c burgers.so
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
