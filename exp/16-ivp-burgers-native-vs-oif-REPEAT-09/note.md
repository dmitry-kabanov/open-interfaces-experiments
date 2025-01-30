# 16 IVP Burgers Performance: Python vs Julia, REPEAT of Exp. 9

## Goal
This is a modified repeat of the Exp. 9.
We compare here performance of Open Interfaces versus native code in Python
and Julia.

The modification is resolutions: I decided to change to resolutions
1600, 6400, 25600 to be consistent with the Exp. 14 "C vs Julia".

## Procedure

To make Python faster,
we optimize Python versions of the right-hand side function using Numba
and rewrite Julia code as loops with macros like `@inbounds` to avoid checks.

I rerun this experiment on `myri`.

For the resolution 25600 `scipy.integrate.dopri5` fails with the default
value of number of steps `nsteps=500`. I set it to `nsteps=1000`.
Actually, there was a bug in `scipy_ode` implementation, due to which
`nsteps` was not passed to the `dopri5` solver after calling
`ivp::set_tolerances` method.

## Results

The results are in the table below:
```
# method                        1600           6400           25600
py-openif-numba-v4              0.122 ± 0.009  1.575 ± 0.005  30.944 ± 0.122
py-native-numba-v3              0.113 ± 0.000  1.573 ± 0.010  30.829 ± 0.121

Julia native
method/resolution               1600           6400           25600
jl-native-v5                    0.067 ± 0.009  0.868 ± 0.004  21.058 ± 0.067

Python via OIF call to `jl_diffeq` (Julia OrdinaryDiffEq.jl)
method/resolution               1600           6400           25600
jl-openif-numba-v4              0.196 ± 0.003  1.466 ± 0.005  28.147 ± 0.040

Python native to SciPy (sanity check)
method/resolution               1600           6400           25600
py-native-numba-v4              0.115 ± 0.000  1.576 ± 0.007  31.134 ± 0.111
```


## Conclusion

The results are basically the same as in the previous experiment (Exp. 9).
