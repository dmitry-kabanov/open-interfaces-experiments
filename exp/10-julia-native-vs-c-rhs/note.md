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

I have switched to Clang 14 and magically, the performance in C became
almost as good as in Julia:
```
Julia, accumulated runtime of 41000 RHS evals, statistics from 30 trials
Problem size is 3201
Julia, v5                        0.310 ± 0.029
Julia, cwrapper-oif              0.330 ± 0.001
Julia, cwrapper-carray           0.328 ± 0.001
Leftmost udot_1 value: -0.0982710900737871
Leftmost udot_2 value: -0.0982710900737871
```

C wrappers are six percent slower than the Julia's magic.
This was the moment where I have stopped optimizing this as it is good enough.

### Solving ODEs

Using the above right-hand sides, I solve the initial-value problem.

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
which shows that with C wrapper it is a 10% penalty for OIF wrapper and
8 percent penalty for C-arrays wrapper.


### Comparison to Python and Julia directly

I use package JuliaCall https://juliapy.github.io/PythonCall.jl/stable/juliacall/
to call Julia from Python.
I first used package `PyJulia` but it seems to be slow (solving ODE at N=3200
takes about 1-1.5 seconds; probably arrays are copied).
Also the package `JuliaCall` is recommended.

This package has a peculiarity that it creates a new Julia environment.
Therefore, one needs to install required packages into this environment.
The thing that worked for me is to create `juliapkg.json` file and then
during import it just add and precompiles the packages.
