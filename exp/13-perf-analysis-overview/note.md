# 13 Performance study overview

## Intro

Here I summarize the performance study of OIF with direct computations.

## Procedure

The results are based on the results of the previous experiments.
In all cases it is based on solving inviscid Burgers' equation
using time integrator Dormand-Prince (Runge--Kutta 5(4))
with tolerances `rtol=1e-6` and `atol=1e-12`.

All simulations are run 30 times and the reported run times are sample means
with uncertainties given by 95% confidential intervals.


## Results


### 1. Python+Numba vs Julia

Here the results are the following:
simply evaluating the right-hand side is faster in Python but time integration
is faster in Julia.

Somehow the right-hand side evaluation in Python + Numba optimizations
is faster than in Julia:
```
Python, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Python + Numba v4           0.229 ± 0.001

Julia, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Julia, v5                   0.301 ± 0.002
```

The number of RHS evals 41000 is chosen while this is what is required
approximately to solve the problem with resolution 3200 points with used
tolerances.

However, solving the actual problem brings completely different picture:
Python is slower, it does not matter if it is native integration or via
Open Interfaces:
```
# method               400          800          1600         3200
jl-native          0.01 ± 0.00  0.02 ± 0.00  0.09 ± 0.00  0.34 ± 0.00
py-native          0.01 ± 0.00  0.04 ± 0.00  0.13 ± 0.00  0.49 ± 0.01
py-openif-scipy    0.02 ± 0.00  0.04 ± 0.00  0.14 ± 0.00  0.50 ± 0.00
py-openif-julia    0.06 ± 0.00  0.11 ± 0.00  0.24 ± 0.00  0.69 ± 0.02
```
One can see from this table that Julia natively (via `OrdinaryDiffEq.jl`)
solves the problem for $N=3200$ in 0.34 seconds,
while Python natively (via SciPy) solves the problem in 0.49 seconds,
and for Python + OIF + SciPy it is only 0.50 seconds,
so basically negligible overhead (probably, because Python + OIF + Python uses
native callbacks instead of wrappers),
while the run time for Python + OIF + Julia is 0.69 seconds.

The main outcome here is that Python + OIF is not that bad in comparison
with Python natively.
However, the run time for Python is worse than for Julia without Open
Interfaces.


### 2. C vs Julia

I have written a right-hand side function in C and make it work almost as fast
as Julia's implementation.

We get the following results:
```
Solving ODEs, statistics from 30 trials
Problem size is 3201
Julia, native                    0.346 ± 0.003
Julia, cwrapper-oif              0.380 ± 0.003
Julia, cwrapper-carray           0.372 ± 0.003
```

The version `cwrapper-oif` is a C function that accepts OIF data types
for arrays (`OIFArrayF64`),
while `cwrapper-carray` works with plain C arrays. We can see that wrapping
plain C arrays with `OIFArrayF64` takes a bit of time but not so much.
Moreover, because the C function was optimized and compiled using Clang,
it gives almost the same performance as if we solve the problem completely
in Julia.
Note that Julia's native right-hand side function was also heavily optimized
with the help of a guy from MIT from Julia's forum:

Then the result of actual time integration are the following
(in table and image forms)

 | Method/N     | 100           | 200           | 400           | 800           | 1600          | 3200          | 6400          |
 | ----------   | -----------   | -----------   | -----------   | -----------   | -----------   | -----------   | -----------   |
 | C + OIF      | 0.032 ± 0.000 | 0.042 ± 0.002 | 0.045 ± 0.002 | 0.062 ± 0.001 | 0.146 ± 0.004 | 0.449 ± 0.005 | 1.881 ± 0.031 |
 | Julia native | 0.001 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.031 ± 0.001 | 0.105 ± 0.002 | 0.409 ± 0.007 | 1.790 ± 0.024 |

```{figure} _assets/perf-c-vs-julia.png

Runtimes in seconds versus grid resolution between `C + OIF` and `Julia
native`, using the same Julia solver in both cases.
```

Here I actually compute to higher resolution than in Python-Julia comparison.

However, for $N = 3200$, we can see that solving the problem
with C + OIF + Julia takes approximately 10% more run time than just using
Julia.


### 3. Old comparison using Python and SUNDIALS

This is a comparison that I did in March 2024.

We solve a reaction-diffusion system (with small diffusion) using Adams-Moulton
scheme from SUNDIALS.
User code is written in Python and does time integration either via OIF
or via direct bindings to SUNDIALS from the package `scikit.odes`.

```{figure} _assets/ivp_cvode_gs_performance.*

Performance comparison of time integration from Python via Open Interfaces 
to a SUNDIALS solver (in C) versus using this solver from Python using
direct bindings provided by the package `scikit.odes`.
```

We can see that actually OIF perform slightly better.
Possible explanation to that is that `scikit.odes` uses Cython for bindings,
which could generate not completely optimal code.

## Conclusions

Performance studies shows that OIF has performance penalty, but it is
about 10% for reasonable workloads.

The byproduct of the studies is that Python simulations are somewhat slower
than Julia simulation, which can be attributed to conversion between NumPy
arrays and Fortran/C data structures that are used with "native" integration
via SciPy or with OIF integration.
