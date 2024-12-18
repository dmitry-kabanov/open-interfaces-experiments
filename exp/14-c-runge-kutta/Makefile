BUILD_TYPE ?= release  # Values: [ debug, release ]

CC := /usr/bin/clang
CPPFLAGS := \
    -I$(OIF_DIR)/oif/include \
	-I$(OIF_DIR)/oif/interfaces/c/include/

CFLAGS := \
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


.PHONY : all
all : burgers.so run_burgers


burgers.so: burgers.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SHLIB_FLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

run_burgers: run_burgers.c burgers.so
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

.PHONY : clean
clean:
	-rm run_burgers burgers.so
