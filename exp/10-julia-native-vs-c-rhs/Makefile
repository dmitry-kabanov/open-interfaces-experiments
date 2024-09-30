CFLAGS := \
    -I$(OIF_DIR)/oif/include \
	-I$(OIF_DIR)/oif/interfaces/c/include/ \
	-O3 \
	-march=native \
	-mtune=native -fomit-frame-pointer -funroll-loops -fno-math-errno -fno-trapping-math -fno-signaling-nans

SHLIB_FLAGS := -shared -fPIC

LDFLAGS := -L$(OIF_DIR)/build
LDLIBS := -loif_c -loif_c_util -loif_c_data_structures -ldl -lm


.PHONY : all
all : burgers.so run_burgers


burgers.so: burgers.c
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

run_burgers: run_burgers.c burgers.so
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

.PHONY : clean
clean:
	-rm run_burgers burgers.so
