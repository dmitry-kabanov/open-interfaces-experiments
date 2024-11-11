# 09 IVP Burgers Performance: optimized RHS

## Goal
In this experiment we compare performance of running numerical solvers natively (directly)
with via Open Interfaces to understand how much performance penalty
is induced.

## Procedure

We optimize Python versions of the right-hand side function using Numba
and rewrite Julia code as loops with macros like `@inbounds` to avoid checks.

The comparison strategy is basically the same as in the previous experiment.

## Results

### Optimization of Python wrapper for callbacks

I profile invokation of Julia sovlers from Python using IPython:
```
%run -p call_ivp_python_jl_diffeq.py jl_diffeq
```
with the following results:
```
  ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    16157   52.057    0.003   86.611    0.005 core.py:200(call)
  2505883    9.757    0.000    9.757    0.000 call_ivp_python_jl_diffeq.py:30(compute_rhs_oif_numba_v3)
  2505732    8.875    0.000   33.859    0.000 core.py:147(wrapper)
      151    7.163    0.047    7.167    0.047 core.py:324(load_impl)
  2505732    5.088    0.000   14.843    0.000 call_ivp_python_jl_diffeq.py:73(compute_rhs_oif)
```
We can see that `wrapper` in `core.py` takes 34 percent of time.
While looking at the code, I have noticed that it has problems such as
`try/catch` block to preserve errors, although it is expensive and it is better
to expect that users do not break function contracts.

After optimizing, the results of profiling, the results are
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    16157   47.565    0.003   80.186    0.005 core.py:200(call)
  2505883    9.423    0.000    9.423    0.000 call_ivp_python_jl_diffeq.py:30(compute_rhs_oif_numba_v3)
  2505732    8.362    0.000   31.969    0.000 core.py:147(wrapper)
  2505732    4.700    0.000   14.122    0.000 call_ivp_python_jl_diffeq.py:73(compute_rhs_oif)
```

### Comparison of pure RHS evaluations

First, we compute aggregated time of evluation RHS 100'000 times.
The following results show how going from straightforward versions
in Python and in Julia to optimized versions gives substantial speedup:
```
Python, accumulated 100000 RHS evals, averaged over 30 trials
Python + NumPy: 5.596 ± 0.031
Python + Numba: 0.690 ± 0.004
Julia, accumulated 100000 RHS evaluations, 30 trials
Julia, plain version: 4.892 ± 0.017
Julia, optim version: 1.211 ± 0.003
```
Somehow, I had first Numba with 1.3 seconds; during the development
it became 0.690 which is super suspicious.

```
---
cat _output/rhs_evals.txt
Python, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Python + NumPy              2.020 ± 0.021
Python + Numba v1           0.542 ± 0.005
Python + Numba v2           0.320 ± 0.005
Python + Numba v3           0.332 ± 0.003
Python + Numba v4           0.229 ± 0.001
Leftmost udot value: -0.0982710900737871

Julia, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Julia, v1                   1.578 ± 0.033
Julia, v2                   0.332 ± 0.002
Julia, v3                   0.300 ± 0.002
Julia, v4                   0.302 ± 0.002
Julia, v5                   0.301 ± 0.002
Leftmost udot value: -0.0982710900737871

---
python run.py

Python native and via OIF: Scipy.integrate.ode.dopri5
# method                        200          400          800          1600         3200
py-openif-numba-v1          0.01 ± 0.00  0.03 ± 0.00  0.06 ± 0.00  0.21 ± 0.00  0.82 ± 0.01
py-openif-numba-v2          0.01 ± 0.00  0.02 ± 0.00  0.05 ± 0.00  0.15 ± 0.00  0.58 ± 0.00
py-openif-numba-v3          0.01 ± 0.00  0.02 ± 0.00  0.05 ± 0.00  0.16 ± 0.00  0.59 ± 0.00
py-openif-numba-v4          0.01 ± 0.00  0.02 ± 0.00  0.04 ± 0.00  0.13 ± 0.00  0.49 ± 0.00
py-openif-numba-v4+wrapper  0.01 ± 0.00  0.02 ± 0.00  0.04 ± 0.00  0.14 ± 0.00  0.50 ± 0.00
py-native-numba-v3          0.01 ± 0.00  0.01 ± 0.00  0.04 ± 0.00  0.13 ± 0.00  0.49 ± 0.01

Julia native
method/resolution               200          400          800          1600         3200
jl-native-v1                0.02 ± 0.00  0.06 ± 0.04  0.17 ± 0.01  0.54 ± 0.07  1.93 ± 0.04
jl-native-v2                0.00 ± 0.00  0.01 ± 0.00  0.02 ± 0.00  0.09 ± 0.00  0.36 ± 0.00
jl-native-v3                0.00 ± 0.00  0.01 ± 0.00  0.02 ± 0.00  0.09 ± 0.00  0.34 ± 0.01
jl-native-v4                0.00 ± 0.00  0.01 ± 0.00  0.02 ± 0.00  0.09 ± 0.00  0.34 ± 0.00
jl-native-v5                0.00 ± 0.00  0.01 ± 0.00  0.02 ± 0.00  0.09 ± 0.00  0.34 ± 0.00

Python via OIF call to `jl_diffeq` (Julia OrdinaryDiffEq.jl)
method/resolution               200          400          800          1600         3200
jl-openif-numba-v4          0.05 ± 0.00  0.06 ± 0.00  0.11 ± 0.00  0.24 ± 0.00  0.69 ± 0.02

Python native to SciPy (sanity check)
method/resolution               200          400          800          1600         3200
py-native-numba-v4          0.01 ± 0.00  0.01 ± 0.00  0.04 ± 0.00  0.14 ± 0.00  0.53 ± 0.01
```

We can see that going from Python to Julia via Open Interfaces is quite slow:
solving the whole thing in Julia takes 0.34 seconds while Python->OIF->Julia
takes 0.68 seconds (2x slower).
And this is although the Numba-optimized RHS is faster than
the Julia's one.

## Conclusion

So, the problem that I have with this experiment is that somehow Numba has
significantly faster RHS evaluation than Julia: ~0.25 vs ~0.33 seconds
for 41000 evaluations.

However, when solving the real problem with time integration (which
requires approximately 41000 RHS evals for the Dormand--Prince RK5(4) method),
with Numba and Python it is about 0.5 seconds (which means that RHS evals
take only 50% of the runtime), while Julia solves it in 0.35 seconds.

So, what is the problem then? Does Python + Fortran dopri5 code together
are so crazy inefficient?
Does Julia somehow optimize the whole thing in a super-duper efficient way?
