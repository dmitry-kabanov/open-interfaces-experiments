#pragma once
#include <stdlib.h>
#include <oif/api.h>

int
rhs_oif_orig(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data);

int
rhs_oif_index_based_max(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data);

int
rhs_oif_simd_max(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data);

