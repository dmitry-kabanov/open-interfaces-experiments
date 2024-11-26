# 12 Performance comparison of C and Julia

After talk with Stephan, we agreed that it is better
to concentrate on comparing running OIF from C with Julia instead of
trying to compare Python and Julia.
The reason is that Python is somehow slow even in native case (without OIF)
and looks pale performance-wise with respect to Julia.


## Procedure

We solve Burgers' equation problem from C using Open Interfaces and
`jl_diffeq` IVP implementation.
We compare runtime with solving the same problem from Julia natively
using `OrdinaryDiffEq.jl` directly.

The right-hand side is exactly the same and written as a C shared library
and compiled using Clang (which Julia also uses internally).

We run simulations with different grid sizes $N$ and run 30 trials
to minimize influences of other computer processes.
The reported runtimes are sample means with 95% confidential interval
based on standard error of the sample mean.

### Correctness

I've tried to make sure that everything, like grid, etc., is computed
in exactly the same fashion in both simulations to make sure that we solve
exactly the same problem.
I expect that both solutions are exactly the same as the same solver is used.
However, somehow solutions agree only to six-seven significant digits.


## Results

Runtimes, in seconds, are the following:

 | Method/N     | 100           | 200           | 400           | 800           | 1600          | 3200          | 6400          |
 | ----------   | -----------   | -----------   | -----------   | -----------   | -----------   | -----------   | -----------   |
 | C + OIF      | 0.032 ± 0.000 | 0.042 ± 0.002 | 0.045 ± 0.002 | 0.062 ± 0.001 | 0.146 ± 0.004 | 0.449 ± 0.005 | 1.881 ± 0.031 |
 | Julia native | 0.001 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.031 ± 0.001 | 0.105 ± 0.002 | 0.409 ± 0.007 | 1.790 ± 0.024 |

```{figure} _assets/perf-c-vs-julia.png

Runtimes in seconds versus grid resolution between `C + OIF` and `Julia
native`, using the same Julia solver in both cases.
```

We can see from this figure that for resolution $N = 6400$, the difference
in runtime is not more than 5% percent.


## Conclusions

We demonstrated in this experiment that invoking Julia solvers from C using
Open Interfaces does not induce a significant performance penalty comparing
to using Julia solver directly from Julia to solve the same problem.
