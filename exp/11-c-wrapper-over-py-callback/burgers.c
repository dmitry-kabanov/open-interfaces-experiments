#include <tgmath.h>
#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/interfaces/ivp.h>


int
rhs_oif(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data)
{
    (void)t;         /* Unused */
    intptr_t N = y->dimensions[0];

    double *u = y->data;
    double *udot = rhs_out->data;

    double dx = *((double *)user_data);
    double dx_inv = 1.0 / dx;

    double local_sound_speed = 0.0;
    for (int i = 0; i < N; ++i) {
        if (local_sound_speed < fabs(u[i])) {
            local_sound_speed = fabs(u[i]);
        }
    }
    double local_ss_rb = fabs(u[0]);
    if (fabs(u[N-1]) > local_ss_rb) {
        local_ss_rb = fabs(u[N-1]);
    }

    double f_cur = 0.5 * pow(u[0], 2);
    double f_hat_lb = 0.5 * (
        (f_cur + 0.5 * pow(u[N-1], 2)) - local_ss_rb * (u[0] - u[N-1])
    );
    double f_hat_prev = f_hat_lb;

    for (int i = 0; i < N - 1; ++i) {
        double f_next = 0.5 * pow(u[i+1], 2);
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
rhs_carray(double t, const double *const y, double *restrict rhs_out, void *restrict user_data, size_t N)
{
    (void)t;         /* Unused */

    const double *const u = y;
    double *udot = rhs_out;

    double dx = *((double *)user_data);
    double dx_inv = 1.0 / dx;

    double local_sound_speed = fabs(u[0]);
    for (int i = 1; i < N; ++i) {
        if (local_sound_speed < fabs(u[i])) {
            local_sound_speed = fabs(u[i]);
        }
    }
    double local_ss_rb = fabs(u[0]);
    if (fabs(u[N-1]) > local_ss_rb) {
        local_ss_rb = fabs(u[N-1]);
    }

    double f_cur = 0.5 * pow(u[0], 2);
    double f_hat_lb = 0.5 * (
        (f_cur + 0.5 * pow(u[N-1], 2)) - local_ss_rb * (u[0] - u[N-1])
    );
    double f_hat_prev = f_hat_lb;

    for (int i = 0; i < N - 1; ++i) {
        double f_next = 0.5 * pow(u[i+1], 2);
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
