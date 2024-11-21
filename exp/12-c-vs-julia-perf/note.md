# 12 Performance comparison of C and Julia

After talk with Stephan on 13 November 2024, we agreed that it is better
to concentrate on comparing running OIF from C with Julia instead of
trying to compare Python and Julia.
The reason is that Python is somehow slow even in native case (without OIF)
and looks pale performance-wise with respect to Julia.


## Procedure

We solve Burgers' equation problem from C using Open Interfaces and
`jl_diffeq` IVP implementation.
We compare runtime with solving the same problem from Julia natively
using `OrdinaryDiffEq.jl` directly.


## Results


## Conclusions
