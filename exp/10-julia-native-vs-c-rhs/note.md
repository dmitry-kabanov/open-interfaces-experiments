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
