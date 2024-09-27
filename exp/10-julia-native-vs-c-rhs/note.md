# Exp10: Comparison of Julia ODE solution: native RHS vs RHS from C

## Goal

We had a discussion with Stephan on 2024-09-27 regarding strange results
that Julia from Python is so slow.
He gave me a suggestion to understand Julia's behavior without OIF but
with wrapper around a C function.
His idea is that maybe the wrapped `$ccall` in Julia is not that cheap.


## Procedure

- We compile a C library that represents Burgers' eq. RHS
- We write a wrapper around C RHS similar to the one in `callback.jl`
- We write a Julia script that invokes Julia's RHS and C-wrapped RHS $N$ times
  and assess the performance differences
- We write a Julia script that invokes `OrdinaryDiffEq.jl` and compares
  performance


## Results

I wrote two versions of the function: the one that works with OIFArrayF64
and another that works directly with C arrays.

I compiled the C library with `-march=native -O3`.

Base performance:
```
Julia, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Julia, v5                        0.318 ± 0.031
Julia, cwrapper-oif              0.442 ± 0.002
Julia, cwrapper-carray           0.439 ± 0.003
Leftmost udot_1 value: -0.0982710900737871
Leftmost udot_2 value: -0.0982710900737871
Leftmost udot_3 value: -0.0982710900737871
```
We can see that the difference between `cwrapper-oif` and `cwrapper-carray` is
negligible.

Then for the version that works with C arrays, I have added `restrict`
and `const` to signature, it helps a bit:
```
Julia, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Julia, v5                        0.314 ± 0.036
Julia, cwrapper-oif              0.441 ± 0.001
Julia, cwrapper-carray           0.394 ± 0.002
Leftmost udot_1 value: -0.0982710900737871
Leftmost udot_2 value: -0.0982710900737871
Leftmost udot_3 value: -0.0982710900737871
```
