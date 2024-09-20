import numpy as np


class BurgersEquationProblem:
    r"""
    Problem class for inviscid Burgers' equation:
    $$
            u_t + 0.5 * (u^2)_x = 0, \quad x \in [0, 2], \quad t \in [0, 2]
    $$
    with initial condition :math:`u(x, 0) = 0.5 - 0.25 * sin(pi * x)`
    and periodic boundary conditions.

    Parameters
    ----------
    N : int
        Grid resolution.
    """

    def __init__(self, N=101):
        self.N = N

        self.x, self.dx = np.linspace(0, 2, num=N + 1, retstep=True)
        self.u0 = 0.5 - 0.25 * np.sin(np.pi * self.x)

        self.cfl = 0.5
        self.dt_max = self.dx * self.cfl

        self.t0 = 0.0
        self.tfinal = 10

        self.udot = np.empty_like(self.u0)
        self.f = np.empty_like(self.u0)
        self.f_hat = np.empty(self.N)
        self.rhs_evals = 0

    def compute_rhs(self, __, u: np.ndarray, udot: np.ndarray, ___) -> None:
        dx = self.dx
        f = self.f
        f_hat = self.f_hat

        f[:] = 0.5 * u**2
        local_ss = np.maximum(np.abs(u[0:-1]), np.abs(u[1:]))
        local_ss = np.max(np.abs(u))
        f_hat[:] = 0.5 * (f[0:-1] + f[1:]) - 0.5 * local_ss * (u[1:] - u[0:-1])
        f_plus = f_hat[1:]
        f_minus = f_hat[0:-1]
        udot[1:-1] = -1.0 / dx * (f_plus - f_minus)

        local_ss_rb = np.maximum(np.abs(u[0]), np.abs(u[-1]))
        f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
        f_lb = f_rb

        udot[+0] = -1.0 / dx * (f_minus[0] - f_lb)
        udot[-1] = -1.0 / dx * (f_rb - f_plus[-1])

        self.rhs_evals += 1

    def compute_rhs_scipy_ode(self, t, u: np.ndarray) -> np.ndarray:
        self.compute_rhs(t, u, self.udot, None)
        return self.udot
