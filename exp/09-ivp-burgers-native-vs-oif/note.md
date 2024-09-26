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

So, the problem that I have with this experiment is that somehow Numba
gives twice faster RHS evaluation than Julia: ~0.25 vs ~0.33 seconds
for 41000 evaluations.

However, when solving the real problem with time integration (which
requires approximately 41000 RHS evals for the Dorman--Prince RK5(4) method),
with Numba and Python it is about 0.5 seconds (which means that RHS evals
take only 50% of the runtime), while Julia solves it in 0.35 seconds!

So, what is the problem then? Does Python + Fortran dopri5 code together
are so crazy inefficient?
Does Julia somehow optimize the whole thing in a super-duper efficient way?

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
------------------------------------------------------------------
Python OIF vs Native
# method                     800          1600         3200
numba-python-native-scipyxx  0.05 ± 0.00  0.16 ± 0.00  0.61 ± 0.00
numba-python-oif-scipy       0.05 ± 0.00  0.15 ± 0.00  0.58 ± 0.00

------------------------------------------------------------------
Julia OIF from Python vs Native

method/resolution            800          1600         3200
@inline step!                0.15 ± 0.02  0.48 ± 0.00  1.83 ± 0.07
@inline solve                0.14 ± 0.00  0.48 ± 0.00  1.84 ± 0.06
@noinline step!              0.14 ± 0.00  0.48 ± 0.00  1.78 ± 0.04
@noinline solve              0.14 ± 0.00  0.50 ± 0.04  1.83 ± 0.07

numba-julia-oif-from-python  0.10 ± 0.00  0.24 ± 0.00  0.72 ± 0.01

------------------------------------------------------------------
Julia DP5 versus Sundials.jl CVODE_Adams

method/resolution           800          1600         3200
julia-DP5                   0.15 ± 0.02  0.47 ± 0.00  1.80 ± 0.04
julia-Sundials-CVODE_Adams  0.26 ± 0.00  0.89 ± 0.04  3.44 ± 0.07

------------------------------------------------------------------
Julia OIF from Python vs Native (inlining is with loop expansion)

method/resolution           800          1600         3200
@inline step!                0.07 ± 0.02  0.19 ± 0.00  0.74 ± 0.01
@inline solve                0.05 ± 0.00  0.19 ± 0.00  0.83 ± 0.07
@noinline step!              0.14 ± 0.00  0.48 ± 0.00  1.78 ± 0.04
@noinline solve              0.14 ± 0.00  0.50 ± 0.04  1.83 ± 0.07
numba-julia-oif-from-python  0.10 ± 0.00  0.24 ± 0.00  0.72 ± 0.01
```

We can see that the results

## Conclusion
