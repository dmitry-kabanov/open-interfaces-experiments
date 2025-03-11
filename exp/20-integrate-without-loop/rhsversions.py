"""Versions of the right-hand side function."""

import numba as nb
import numpy as np


def compute_rhs_oif_numpy(__, u: np.ndarray, udot: np.ndarray, p) -> None:
    (dx,) = p

    f = 0.5 * u**2
    local_ss = np.max(np.abs(u))

    f_hat = 0.5 * (f[0:-1] + f[1:]) - 0.5 * local_ss * (u[1:] - u[0:-1])
    f_plus = f_hat[1:]
    f_minus = f_hat[0:-1]
    udot[1:-1] = -1.0 / dx * (f_plus - f_minus)

    local_ss_rb = np.maximum(np.abs(u[0]), np.abs(u[-1]))
    f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_lb = f_rb

    udot[+0] = -1.0 / dx * (f_minus[0] - f_lb)
    udot[-1] = -1.0 / dx * (f_rb - f_plus[-1])


@nb.jit
def compute_rhs_oif_numba_v1(__, u: np.ndarray, udot: np.ndarray, p) -> None:
    (dx,) = p

    f = 0.5 * u**2
    local_ss = np.max(np.abs(u))

    f_hat = 0.5 * (f[0:-1] + f[1:]) - 0.5 * local_ss * (u[1:] - u[0:-1])
    f_plus = f_hat[1:]
    f_minus = f_hat[0:-1]
    udot[1:-1] = -1.0 / dx * (f_plus - f_minus)

    local_ss_rb = np.maximum(np.abs(u[0]), np.abs(u[-1]))
    f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_lb = f_rb

    udot[+0] = -1.0 / dx * (f_minus[0] - f_lb)
    udot[-1] = -1.0 / dx * (f_rb - f_plus[-1])


@nb.jit
def compute_rhs_oif_numba_v2(__, u: np.ndarray, udot: np.ndarray, p) -> None:
    (dx,) = p

    N = u.shape[0]

    f = np.empty(N)
    for i in range(N):
        f[i] = 0.5 * u[i] ** 2

    local_ss = 0.0
    for i in range(N - 1):
        cand = abs(u[i])
        if cand > local_ss:
            local_ss = cand

    f_hat = np.empty(N - 1)
    for i in range(N - 1):
        f_hat[i] = 0.5 * (f[i] + f[i + 1]) - 0.5 * local_ss * (u[i + 1] - u[i])

    for i in range(1, N - 1):
        udot[i] = -1.0 / dx * (f_hat[i] - f_hat[i - 1])

    local_ss_rb = max(abs(u[0]), abs(u[-1]))
    f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_lb = f_rb

    udot[0] = -1.0 / dx * (f_hat[0] - f_lb)
    udot[-1] = -1.0 / dx * (f_rb - f_hat[-1])


def compute_rhs_fused_loops(__, u: np.ndarray, udot: np.ndarray, p) -> None:
    (dx,) = p
    N = u.shape[0]

    local_ss = 0.0
    for i in range(N - 1):
        cand = abs(u[i])
        if cand > local_ss:
            local_ss = cand
    local_ss_rb = max(abs(u[0]), abs(u[-1]))

    dx_inv = 1.0 / dx

    f_cur = 0.5 * u[0] ** 2
    f_hat_lb = 0.5 * (f_cur + 0.5 * u[-1] ** 2) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_hat_prev = f_hat_lb
    for i in range(N - 1):
        f_next = 0.5 * u[i + 1] ** 2
        f_hat_cur = 0.5 * ((f_cur + f_next) - local_ss * (u[i + 1] - u[i]))
        udot[i] = dx_inv * (f_hat_prev - f_hat_cur)
        f_hat_prev, f_cur = f_hat_cur, f_next

    f_hat_rb = f_hat_lb
    udot[-1] = dx_inv * (f_hat_prev - f_hat_rb)


compute_rhs_oif_numba_v3 = nb.jit(
    nb.types.void(nb.float64, nb.float64[:], nb.float64[:], nb.typeof((3.14,))),
    boundscheck=False,
    nogil=True,
)(compute_rhs_fused_loops)


# For some mysterious reason, removing signature improves performance.
compute_rhs_oif_numba_v4 = nb.jit(
    boundscheck=False,
    nogil=True,
)(compute_rhs_fused_loops)
