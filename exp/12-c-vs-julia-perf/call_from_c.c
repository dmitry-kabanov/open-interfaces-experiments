/**
 * Run OIF for Burgers' equation and measure performance of time integration.
 */
#include <assert.h>
#include <tgmath.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/interfaces/ivp.h>

#include "burgers.h"

char *
parse_impl(int argc, char *argv[])
{
    if (argc == 1) {
        return "scipy_ode";
    }
    else {
        if ((strcmp(argv[1], "scipy_ode") == 0) || (strcmp(argv[1], "sundials_cvode") == 0) ||
            (strcmp(argv[1], "jl_diffeq") == 0)) {
            return argv[1];
        }
        else {
            fprintf(stderr, "USAGE: %s [scipy_ode | sundials_cvode | jl_diffeq]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

char *
parse_output_filename(int argc, char *argv[])
{
    if (argc < 3) {
        return "_output/solution.txt";
    }
    else {
        return argv[2];
    }
}

int
parse_resolution(int argc, char *argv[])
{
    if (argc < 4) {
        return 3200;
    }
    else {
        return atoi(argv[3]);
    }
}

static int
compute_initial_condition_(size_t N, OIFArrayF64 *u0, OIFArrayF64 *grid, double *dx,
                           double *dt_max)
{
    double a = 0.0;
    double b = 2.0;
    double *x = grid->data;
    *dx = (b - a) / N;

    for (int i = 0; i <= N; ++i) {
        x[i] = a + i * (*dx);
    }

    for (int i = 0; i <= N; ++i) {
        u0->data[i] = 0.5 - 0.25 * sin(M_PI * x[i]);
    }

    double cfl = 0.5;
    *dt_max = cfl * (*dx);

    return 0;
}
int
main(int argc, char *argv[])
{
    int retval = 0;
    char *impl = parse_impl(argc, argv);
    const char *output_filename = parse_output_filename(argc, argv);
    const int N = parse_resolution(argc, argv);
    const int T = 101;
    printf("Calling from C an open interface for solving y'(t) = f(t, y)\n");
    printf("where the system comes from inviscid 1D Burgers' equation\n");
    printf("Implementation: %s\n", impl);
    printf("Resolution: %d\n", N);
    printf("Number of time steps: %d\n", T);

    double t0 = 0.0;
    double t_final = 10.0;
    OIFArrayF64 *y0 = oif_create_array_f64(1, (intptr_t[1]){N});
    OIFArrayF64 *y0 = oif_create_array_f64(1, (intptr_t[1]){N + 1});
    // Solution vector.
    OIFArrayF64 *y = oif_create_array_f64(1, (intptr_t[1]){N + 1});
    OIFArrayF64 *y = oif_create_array_f64(1, (intptr_t[1]){N});
    // Grid
    OIFArrayF64 *grid = oif_create_array_f64(1, (intptr_t[1]){N + 1});
    OIFArrayF64 *grid = oif_create_array_f64(1, (intptr_t[1]){N});
    double dx;
    double dt_max;
    int status = 1;  // Aux variable to check for errors.

    status = compute_initial_condition_(N, y0, grid, &dx, &dt_max);
    assert(status == 0);

    ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    if (implh == OIF_IMPL_INIT_ERROR) {
        fprintf(stderr, "Error during implementation initialization. Cannot proceed\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }

    status = oif_ivp_set_initial_value(implh, y0, t0);
    if (status) {
        fprintf(stderr, "oif_ivp_set_set_initial_value returned error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    status = oif_ivp_set_user_data(implh, &dx);
    if (status) {
        fprintf(stderr, "oif_ivp_set_user_data return error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    status = oif_ivp_set_rhs_fn(implh, rhs_oif);
    if (status) {
        fprintf(stderr, "oif_ivp_set_rhs_fn returned error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }

    status = oif_ivp_set_tolerances(implh, 1e-8, 1e-12);
    assert(status == 0);

    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "dense", 0);
    oif_config_dict_add_int(dict, "save_everystep", 0);

    if (strcmp(impl, "scipy_ode") == 0) {
        status = oif_ivp_set_integrator(implh, "dopri5", NULL);
    }
    else if (strcmp(impl, "jl_diffeq") == 0) {
        status = oif_ivp_set_integrator(implh, "DP5", NULL);
        /* status = oif_ivp_set_integrator(implh, "DP5", dict); */
    }
    assert(status == 0);
    /* double t = 0.0001; */
    /* status = oif_ivp_integrate(implh, t, y); */
    assert(status == 0);

    clock_t tic = clock();
    // Time step.
    double dt = t_final / T;
    for (int i = 0; i <= T; ++i) {
        double t = t0 + (i + 1) * dt;
        if (t > t_final) {
            t = t_final;
        }
        status = oif_ivp_integrate(implh, t, y);
        if (status) {
            fprintf(stderr, "oif_ivp_integrate returned error\n");
            retval = EXIT_FAILURE;
            goto cleanup;
        }
    }
    clock_t toc = clock();
    printf("Elapsed time = %.6f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    /* printf("Number of right-hand side evaluations = %d\n", N_RHS_EVALS); */

    FILE *fp = fopen(output_filename, "w+e");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file '%s' for writing\n", output_filename);
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    for (int i = 0; i < N; ++i) {
        fprintf(fp, "%.8f %.8f\n", grid->data[i], y->data[i]);
    }
    fclose(fp);
    printf("Solution was written to file `%s`\n", output_filename);

cleanup:
    oif_free_array_f64(y0);
    oif_free_array_f64(y);
    oif_free_array_f64(grid);

    return retval;
}
