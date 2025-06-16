#include <tgmath.h>
#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/interfaces/ivp.h>
#include <immintrin.h>


int
rhs_oif_orig(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data)
{
    (void)t;         /* Unused */
    const intptr_t N = y->dimensions[0];

    const double * const restrict u = y->data;
    double * restrict udot = rhs_out->data;

    double dx = *((double *)user_data);
    double dx_inv = 1.0 / dx;

    double local_sound_speed = 0.0;
    for (intptr_t i = 0; i < N; ++i) {
        double val = fabs(u[i]);
        if (val > local_sound_speed) {
            local_sound_speed = val;
        }
    }

    // Boundary cases
    double local_ss_rb = fmax(fabs(u[0]), fabs(u[N-1]));

    double f_cur = 0.5 * (u[0] * u[0]);
    double f_hat_lb = 0.5 * (
        (f_cur + 0.5 * (u[N-1] * u[N-1])) - local_ss_rb * (u[0] - u[N-1])
    );
    double f_hat_prev = f_hat_lb;

    for (intptr_t i = 0; i < N - 1; ++i) {
        /* double f_next = 0.5 * pow(u[i+1], 2); */
        double f_next = 0.5 * (u[i+1] * u[i+1]);
        double f_hat_cur = 0.5 * (
            (f_cur + f_next) - local_sound_speed * (u[i + 1] - u[i])
        );
        udot[i] = dx_inv * (f_hat_prev - f_hat_cur);
        f_hat_prev = f_hat_cur;
        f_cur = f_next;
    }
    udot[N - 1] = dx_inv * (f_hat_prev - f_hat_lb);

    return 0;
}



int
rhs_oif_index_based_max(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data)
{
    (void)t;         /* Unused */
    const intptr_t N = y->dimensions[0];

    const double * const restrict u = y->data;
    double * restrict udot = rhs_out->data;

    double dx = *((double *)user_data);
    double dx_inv = 1.0 / dx;

    // Index-based search, works faster than the value-based one above.
    intptr_t max_i = 0;
    for (intptr_t i = 1; i < N; ++i) {
        if (fabs(u[i]) > u[max_i]) {
            max_i = i;
        }
    }
    double local_sound_speed = fabs(u[max_i]);

    // Boundary cases
    double local_ss_rb = fmax(fabs(u[0]), fabs(u[N-1]));

    double f_cur = 0.5 * (u[0] * u[0]);
    double f_hat_lb = 0.5 * (
        (f_cur + 0.5 * (u[N-1] * u[N-1])) - local_ss_rb * (u[0] - u[N-1])
    );
    double f_hat_prev = f_hat_lb;

    for (intptr_t i = 0; i < N - 1; ++i) {
        /* double f_next = 0.5 * pow(u[i+1], 2); */
        double f_next = 0.5 * (u[i+1] * u[i+1]);
        double f_hat_cur = 0.5 * (
            (f_cur + f_next) - local_sound_speed * (u[i + 1] - u[i])
        );
        udot[i] = dx_inv * (f_hat_prev - f_hat_cur);
        f_hat_prev = f_hat_cur;
        f_cur = f_next;
    }
    udot[N - 1] = dx_inv * (f_hat_prev - f_hat_lb);

    return 0;
}


int
rhs_oif_simd_max(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data)
{
    (void)t;         /* Unused */
    const intptr_t N = y->dimensions[0];

    const double * const restrict u = y->data;
    double * restrict udot = rhs_out->data;

    double dx = *((double *)user_data);
    double dx_inv = 1.0 / dx;

    // SIMD-based max absolute value computation
    // Was written with the help of ChatGPT 4o in GitHub Copilot.
    // It is FASTER than Julia's code (see rhsversions.jl, function v5).
    __m256d max_vec = _mm256_setzero_pd();
    for (intptr_t i = 0; i < N; i += 4) {
        __m256d vec = _mm256_loadu_pd(&u[i]);
        __m256d abs_vec = _mm256_andnot_pd(_mm256_set1_pd(-0.0), vec);
        max_vec = _mm256_max_pd(max_vec, abs_vec);
    }
    double max_vals[4];
    _mm256_storeu_pd(max_vals, max_vec);
    double local_sound_speed = fmax(fmax(max_vals[0], max_vals[1]), fmax(max_vals[2], max_vals[3]));

    // Boundary cases
    double local_ss_rb = fmax(fabs(u[0]), fabs(u[N-1]));

    double f_cur = 0.5 * (u[0] * u[0]);
    double f_hat_lb = 0.5 * (
        (f_cur + 0.5 * (u[N-1] * u[N-1])) - local_ss_rb * (u[0] - u[N-1])
    );
    double f_hat_prev = f_hat_lb;

    for (intptr_t i = 0; i < N - 1; ++i) {
        /* double f_next = 0.5 * pow(u[i+1], 2); */
        double f_next = 0.5 * (u[i+1] * u[i+1]);
        double f_hat_cur = 0.5 * (
            (f_cur + f_next) - local_sound_speed * (u[i + 1] - u[i])
        );
        udot[i] = dx_inv * (f_hat_prev - f_hat_cur);
        f_hat_prev = f_hat_cur;
        f_cur = f_next;
    }
    udot[N - 1] = dx_inv * (f_hat_prev - f_hat_lb);

    return 0;
}
