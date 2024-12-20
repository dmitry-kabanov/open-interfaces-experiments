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

enum {
    N_TRIALS = 30,
};


int
parse_resolution(int argc, char *argv[])
{
    if (argc < 2) {
        return 3200;
    }
    else {
        return atoi(argv[1]);
    }
}

static double
compute_mean(double *values, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) {
        mean += values[i];
    }
    mean /= n;

    return mean;
}

static double
compute_ci(double *values, int n, double mean) {
    double var = 0.0;

    for (int i = 0; i < n; ++i) {
        var = (values[i] - mean) * (values[i] - mean);
    }
    var /= (n - 1);

    double sem = sqrt(var / n);
    double ci = 2 * sem;
    return ci;
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
benchmark_one_run(
    const char *impl, const char *solution_filename, int N, bool save_solution, double *p_runtime)
{
    int retval = -1;
    double t0 = 0.0;
    double t_final = 10.0;
    const int T = 101;  // Number of time steps.
    OIFArrayF64 *y0 = oif_create_array_f64(1, (intptr_t[1]){N + 1});
    // Solution vector.
    OIFArrayF64 *y = oif_create_array_f64(1, (intptr_t[1]){N + 1});
    // Grid
    OIFArrayF64 *grid = oif_create_array_f64(1, (intptr_t[1]){N + 1});
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

    status = oif_ivp_set_tolerances(implh, 1e-6, 1e-12);
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
    *p_runtime = (double)(toc - tic) / CLOCKS_PER_SEC;
    printf("Elapsed time = %.3f seconds\n", *p_runtime);
    /* printf("Number of right-hand side evaluations = %d\n", N_RHS_EVALS); */

    if (save_solution) {
        FILE *fp = fopen(solution_filename, "w+e");
        if (fp == NULL) {
            fprintf(stderr, "Could not open file '%s' for writing\n", solution_filename);
            retval = EXIT_FAILURE;
            goto cleanup;
        }
        for (int i = 0; i <= N; ++i) {
            fprintf(fp, "%.16f %.16f\n", grid->data[i], y->data[i]);
        }
        fclose(fp);
        printf("Solution was written to file `%s`\n", solution_filename);
    }

    retval = 0;

cleanup:
    oif_free_array_f64(y0);
    oif_free_array_f64(y);
    oif_free_array_f64(grid);

    return retval;

}

int
main(int argc, char *argv[])
{
    int retval = -1;
    const char *impl = "jl_diffeq";
    const int N = parse_resolution(argc, argv);
    char solution_filename[1024];
    char fmt[] = "_output/N=%04d/solution-c.txt";
    int nbytes_written = snprintf(solution_filename, sizeof solution_filename, fmt, N);
    if (nbytes_written < 0 || nbytes_written >= sizeof solution_filename) {
        fprintf(
            stderr,
            "[main] Cannot format `solution_filename`: "
            "need %zu bytes, but have only %zu bytes\n",
            strlen(fmt) + 1,
            sizeof solution_filename
        );
        goto finally;
    }
    bool save_solution = false;

    // ========================================================================
    // Allocate resources.
    double *runtimes = NULL;
    FILE *fh = NULL;

    char runtimes_filename[512];
    char runtimes_fmt[] = "_output/N=%04d/runtimes-c.txt";
    nbytes_written = snprintf(runtimes_filename, sizeof runtimes_filename, runtimes_fmt, N);
    if (nbytes_written < 0 || nbytes_written >= sizeof runtimes_filename) {
        fprintf(
            stderr,
            "[main] Cannot format `runtimes_filename`: "
            "need %zu bytes, but have only %zu bytes\n",
            strlen(runtimes_fmt) + 1,
            sizeof runtimes_filename
        );
        goto finally;
    }

    fh = fopen(runtimes_filename, "w");
    if (fh == NULL) {
        fprintf(stderr,
            "[main] Could not open file to write runtimes '%s'\n",
            runtimes_filename);
        goto clean;
    }

    runtimes = malloc(sizeof(*runtimes) * N_TRIALS);
    if (runtimes == NULL) {
        fprintf(stderr, "[main] Could not allocate memory for the runtimes array\n");
        goto finally;
    }
    // ========================================================================

    printf("Calling from C an open interface for solving y'(t) = f(t, y)\n");
    printf("where the system comes from inviscid 1D Burgers' equation\n");
    printf("Implementation: %s\n", impl);
    printf("Solution filename: %s\n", solution_filename);
    printf("Resolution: %d\n", N);

    printf("=== BEGIN warmup\n");
    benchmark_one_run(impl, solution_filename, N, save_solution, &runtimes[0]);
    printf("=== END warmup\n");

    for (int i = 0; i < N_TRIALS; ++i) {
        if (i == N_TRIALS - 1) {
            save_solution = true;
        }
        int status = benchmark_one_run(impl, solution_filename, N, save_solution, &runtimes[i]);
        assert(status == 0);
    }

    double mean_runtime = compute_mean(runtimes, N_TRIALS);
    double ci = compute_ci(runtimes, N_TRIALS, mean_runtime);

    printf("Runtime, sec: %.3f ± %.12f\n", mean_runtime, ci);
    retval = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        fprintf(fh, "%.16f\n", runtimes[i]);
    }

clean:
    if (fh != NULL) {
        fclose(fh);
    }
    if (runtimes != NULL) {
        free(runtimes);
    }

finally:
    return retval;
}
