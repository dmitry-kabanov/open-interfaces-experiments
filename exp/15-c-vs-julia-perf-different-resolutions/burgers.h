#pragma once
#include <stdlib.h>
#include <oif/api.h>

int
rhs_oif(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data);

int
rhs_carray(
    double t, const double *const y, double *restrict rhs_out, void *restrict user_data, size_t N);

