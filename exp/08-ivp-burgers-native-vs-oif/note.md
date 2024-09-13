(08-ivp-burgers-perf=)
# 08 IVP Burgers Performance: more tests

## Goal
In this experiment we compare performance of running numerical solvers natively (directly)
with via Open Interfaces to understand how much performance penalty
is induced.

## Procedure

Procedure is written in the previous experiment (#07-ivp-burgers-perf).




## Results

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
