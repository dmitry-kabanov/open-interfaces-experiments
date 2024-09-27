CFLAGS := -shared -fPIC \
    -I$(OIF_DIR)/oif/include \
	-I$(OIF_DIR)/oif/interfaces/c/include/ \
	-O3

LDFLAGS := -L$(OIF_DIR)/build
LDLIBS := -loif_c -loif_c_util -loif_c_data_structures -lm


.PHONY : all
all : burgers.so


burgers.so: burgers.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
