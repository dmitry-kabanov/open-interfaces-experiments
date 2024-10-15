#pragma once
#include <stdlib.h>


int
rhs_carray(
    double t, const double *const y, double *restrict rhs_out, void *restrict user_data, size_t N);

