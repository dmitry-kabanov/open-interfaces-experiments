#include <assert.h>
#include <stdio.h>
#include <tgmath.h>
#include <time.h>

#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/interfaces/ivp.h>
#include "burgers.h"

static int
compute_initial_condition_(size_t N, OIFArrayF64 *u0, OIFArrayF64 *grid, double *dx,
                           double *dt_max)
{
    double a = 0.0;
    double b = 2.0;
    double *x = grid->data;
    *dx = (b - a) / N;

    for (size_t i = 0; i <= N; ++i) {
        x[i] = a + i * (*dx);
    }

    for (size_t i = 0; i <= N; ++i) {
        u0->data[i] = 0.5 - 0.25 * sin(M_PI * x[i]);
    }

    double cfl = 0.5;
    *dt_max = cfl * (*dx);

    return 0;
}

static int
parse_resolution_(int argc, char *argv[])
{
    if (argc < 2) {
        return 3200;
    }

    return atoi(argv[1]);
}

int
main(int argc, char *argv[])
{
    int retval = -1;
    double t0 = 0.0;
    double t_final = 10.0;
    int N = parse_resolution_(argc, argv);
    printf("N = %d\n", N);
    OIFArrayF64 *y0 = oif_create_array_f64(1, (intptr_t[1]){N + 1});
    // Solution vector.
    OIFArrayF64 *y = oif_create_array_f64(1, (intptr_t[1]){N + 1});
    // Grid
    OIFArrayF64 *grid = oif_create_array_f64(1, (intptr_t[1]){N + 1});
    double dx;
    double dt_max;
    int T = 101;
    int status = 1; // Aux variable to check for errors.

    status = compute_initial_condition_(N, y0, grid, &dx, &dt_max);
    assert(status == 0);

    const char impl[] = "dopri5c";
    ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    if (implh == OIF_IMPL_INIT_ERROR)
    {
        fprintf(stderr, "Error during implementation initialization. Cannot proceed\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }

    status = oif_ivp_set_initial_value(implh, y0, t0);
    if (status)
    {
        fprintf(stderr, "oif_ivp_set_set_initial_value returned error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    status = oif_ivp_set_user_data(implh, &dx);
    if (status)
    {
        fprintf(stderr, "oif_ivp_set_user_data return error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    status = oif_ivp_set_rhs_fn(implh, rhs_oif);
    if (status)
    {
        fprintf(stderr, "oif_ivp_set_rhs_fn returned error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }

    status = oif_ivp_set_tolerances(implh, 1e-6, 1e-12);
    assert(status == 0);

    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "dense", 0);
    oif_config_dict_add_int(dict, "save_everystep", 0);

    /* double t = 0.0001; */
    /* status = oif_ivp_integrate(implh, t, y); */

    double dt = (t_final - t0) / T;

    clock_t tic = clock();
    // Time step.
    for (int i = 0; i < T; ++i)
    {
        double t = t0 + (i + 1) * dt;
        if (t > t_final)
        {
            t = t_final;
        }
        status = oif_ivp_integrate(implh, t, y);
        if (status)
        {
            fprintf(stderr, "oif_ivp_integrate returned error\n");
            retval = EXIT_FAILURE;
            goto cleanup;
        }
    }
    clock_t toc = clock();
    printf("Elapsed time = %.6f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    oif_ivp_print_stats(implh);

    const char output_filename[] = "_output/solution_dopri5c_oif.txt";
    FILE *fp = fopen(output_filename, "w+e");
    if (fp == NULL)
    {
        fprintf(stderr, "Could not open file '%s' for writing\n", output_filename);
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    for (int i = 0; i <= N; ++i)
    {
        fprintf(fp, "%.8f %.8f\n", grid->data[i], y->data[i]);
    }
    fclose(fp);
    printf("Solution was written to file `%s`\n", output_filename);
    retval = EXIT_SUCCESS;

cleanup:
    oif_free_array_f64(y0);
    oif_free_array_f64(y);
    oif_free_array_f64(grid);

    return retval;
}
