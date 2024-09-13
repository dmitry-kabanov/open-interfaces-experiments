(07-ivp-burgers-perf=)
# 07 IVP Burgers Performance

## Goal
In this experiment we compare performance of running numerical solvers natively (directly)
with via Open Interfaces to understand how much performance penalty
is induced.

## Procedure

We use inviscid Burgers' equation as a problem that is integrated
in time by solvers for ODEs
to conduct the experiment.

First, we compare the performance using Python client which runs
SciPy `ode.dopri5` time integrator directly and via OIF.

Second, we compare the performance of running `OrdinaryDiffEq.jl` Julia package
from Julia itself and from Python client via OIF.

For performance study, we run 5 trials with 30 runs in each.
For each run we measure the performance time only for the integration block
itself (without setting initial conditions, etc.).

The reported runtime is the mean over the sample of 30 runs.
The uncertainty is computed as the 95% confidence interval, which is
the standard error of the mean multiplied by 2.0.

More precisely, let $r_1, \dots, r_N$ with $N=30$ be a sample of runtimes.
Then the reported runtime result is
```{math}
    \bar r \pm 2 \sigma_{\bar r}
```
where
\begin{align}
    \bar r = \frac1N \sum_{i=1}^N r_i \quad \mathrm{\ and\ } \quad
    \sigma_{\bar r} = \sqrt{\frac{\sum_{i=1}^N \left(r_i - \bar r \right)^2}{N (N-1)}}
\end{align}


## Results

Results of running the simulations in Python are the following:

```{code}
Python OIF vs Native
# method  800          1600         3200
native    0.35 ± 0.00  0.85 ± 0.00  2.34 ± 0.02
oif       0.37 ± 0.01  0.87 ± 0.01  2.38 ± 0.02

```

We can see that the results are practically identical.
I should note that when going Python to Python, callbacks are native,
that is, there is no C wrapper that could affect the performance.

Results of running the simulations completely natively in Julia
versus calling Julia from Python via OIF are the following:

```
method/resolution  800              1600             3200
step!              0.1521 ± 0.0240  0.4681 ± 0.0021  1.7583 ± 0.0454
solve              0.1347 ± 0.0011  0.4619 ± 0.0015  1.7931 ± 0.0600
oif from python    0.4532 ± 0.0042  1.0254 ± 0.0092  2.5669 ± 0.0190
```

Lines `step!` and `solve` are just two ways of calling Julia: either
explicitly set each next time step, or let Julia decide, on which time points
to return the solution.

We can see that using Julia via OIF gives 2x performance penalty for small
resolutions, and about 44% penalty for the largest resolution 3200 points.

Note that when calling Julia from Python, the callback is implemented
in NumPy and wrapped as a C function.

## Conclusion

After discussion with Stephan on 11 Sep 2024, we have decided the following
experiments to augment these results:

- Compare simply RHS evaluation in Python and Julia
- Try Numba on RHS evaluation in Python
- Use `@noinline` in Julia to see how it affects the performance
- Write a C function and wrap it in both Python and Julia
- Write Julia script that compares performance of a Julia native solver
  with `Sundials.jl`

Ideally, the client from Julia is required so that the results can be
really compared correctly.
