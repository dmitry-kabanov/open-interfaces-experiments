# 10 Comparison of Julia ODE solution: native RHS vs RHS from C

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

### Writing C wrapper and making sure it is fast

I wrote two versions of the function: the one that works with `OIFArrayF64`
and another that works directly with C arrays:
they are denoted as `cwrapper-oif-array` and `cwrapper-c-array`, respectively.

Note the base performance and optimization are not strictly reproducible
as I have modified the code in-place (as well as the change of the compiler).

To run the RHS runtime comparison:
```
julia call_rhs_eval_julia.jl
```

#### Base performance

I have a version in C quite similar to the Julia version:
manually optimized dense code that passes through the grid only two times
(once to find the sound speed and the second to update fluxes).

I compiled the C library with `-march=native -O3`.
```
Julia, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Julia, v5                        0.318 ± 0.031
Julia, cwrapper-oif-array        0.442 ± 0.002
Julia, cwrapper-c-array          0.439 ± 0.003
Leftmost udot_1 value: -0.0982710900737871
Leftmost udot_2 value: -0.0982710900737871
Leftmost udot_3 value: -0.0982710900737871
```
We can see that the difference between `cwrapper-oif` and `cwrapper-carray` is
negligible.

#### Optimization 1

Then for the version that works with C arrays, I have added `restrict`
and `const` to signature, it helps a bit:
```
Julia, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Julia, v5                        0.314 ± 0.036
Julia, cwrapper-oif-array        0.441 ± 0.001
Julia, cwrapper-c-array          0.394 ± 0.002
Leftmost udot_1 value: -0.0982710900737871
Leftmost udot_2 value: -0.0982710900737871
Leftmost udot_3 value: -0.0982710900737871
```

#### Optimization 2

I have switched to Clang 14 and magically, the performance in C became
almost as good as in Julia:
```
Julia, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Julia, v5                        0.310 ± 0.029
Julia, cwrapper-oif-array        0.330 ± 0.001
Julia, cwrapper-c-array          0.328 ± 0.001
Leftmost udot_1 value: -0.0982710900737871
Leftmost udot_2 value: -0.0982710900737871
```

C wrappers are six percent slower than the Julia's magic.
This was the moment where I have stopped optimizing this as it is good enough.

### OrdinaryDiffEq.jl: comparison of native RHS and C versions

Using the above right-hand sides, I solve the initial-value problem:
```
julia call_ivp_julia.jl
```

I wrap C-wrappers one more time to match the signature expected by the
`OrdinaryDiffEq.jl`.
So it is basically now a Julia function that calls a Julia function that calls
a C function - one less step than with OIF+Python where the C function
is calling the underlying Python function.

Each problem is solved 30 times to compute statistics.
Each right-hand side was called once before the trial to compile.

The results are the following:
```
Solving ODEs, statistics from 30 trials
Problem size is 3201
Julia, v5                        0.346 ± 0.003
Julia, cwrapper-oif              0.380 ± 0.003
Julia, cwrapper-carray           0.372 ± 0.003
```
which shows that with C wrapper it is a 10% penalty for the version
with `OIFArrayF64` data types and eight percent penalty for the version
with C arrays.


## Conclusion

The main obstacle to getting good performance right now is that somehow
translation of arrays between languages takes large portion of runtime---
simple operation like getting a pointer to a data buffer is comparable
to computations of the right-hand side.
