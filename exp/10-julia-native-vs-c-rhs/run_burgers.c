#include <assert.h>
#include <stdio.h>
#include <tgmath.h>
#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/interfaces/ivp.h>


static void __attribute__((constructor)) debug_init(void) {
    printf("DEBUG: Program starting\n");
    fflush(stdout);
}


int
rhs_carray(
    double t, const double *const y, double *restrict rhs_out, void *restrict user_data, size_t N);

static int
compute_initial_condition_(size_t N, OIFArrayF64 *u0, OIFArrayF64 *grid, double *dx,
                           double *dt_max)
{
    double a = 0.0;
    double b = 2.0;
    double *x = grid->data;
    *dx = (b - a) / N;

    for (int i = 0; i < N; ++i) {
        x[i] = a + i * (*dx);
    }

    for (int i = 0; i < N; ++i) {
        u0->data[i] = 0.5 - 0.25 * sin(M_PI * x[i]);
    }

    double cfl = 0.5;
    *dt_max = cfl * (*dx);

    return 0;
}


int main() {
    printf("HERE 1\n");
    double t0 = 0.0;
    size_t N = 32001;
    OIFArrayF64 *y0 = oif_create_array_f64(1, (intptr_t[1]){N});
    // Solution vector.
    OIFArrayF64 *ydot = oif_create_array_f64(1, (intptr_t[1]){N});
    // Grid
    OIFArrayF64 *grid = oif_create_array_f64(1, (intptr_t[1]){N});
    printf("HERE 2\n");
    double dx;
    double dt_max;
    int status = 1;  // Aux variable to check for errors.

    status = compute_initial_condition_(N, y0, grid, &dx, &dt_max);
    printf("HERE 3\n");
    assert(status == 0);

    rhs_carray(t0, y0->data, ydot->data, &dx, N);

    printf("ydot[0] = %.f\n", ydot->data[0]);
}
