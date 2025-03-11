# 20 IVP Burgers Performance: Python vs Julia, without loop

## Goal
This is a modified version of the Exp. 16.
We compare here performance of Open Interfaces versus native code in Python
and Julia.

**What is modified wrt Exp.16:**
we integrate without a loop, that is, we invoke `integrate`
only once from `t0` to `tfinal`.
The hypothesis that we want to test here is
that the performance in Python is significantly worse than in Julia
due to the overhead of the loop, in which we invoke `integrate`
for a set of time points.

## Procedure

I run this experiment on `myri`.

## Results

### Only RHS evaluation


The results demonstrate that RHS evaluation in Python is a bit faster
than in Julia:
```
Python, accumulated runtime of 10000 RHS evals, statistics from 29 trials
Problem size is 6401
Python + NumPy                   0.483 ± 0.003
Python + Numba v1                0.317 ± 0.000
Python + Numba v2                0.157 ± 0.000
Python + Numba v3                0.172 ± 0.001
Python + Numba v4                0.120 ± 0.000
Leftmost udot value: -0.0982229460661621

Julia, accumulated runtime of 10000 RHS evals, statistics from 30 trials
Problem size is 6401
Julia, v1                        0.711 ± 0.008
Julia, v2                        0.178 ± 0.004
Julia, v3                        0.127 ± 0.002
Julia, v4                        0.134 ± 0.006
Julia, v5                        0.134 ± 0.002
Leftmost udot value: -0.0982229460661621
```
**Note:** Unfortunately, I could not reproduce the results that are in the paper:
0.116 for Python and 0.122 for Julia, in spite of very tight confidence
intervals.

### Integration

When we do the actual integration, we see that Julia natively is faster

I have redone the logic of labels:

- First segment (py or jl) specifies user's language
- Second segment (numba-v4 or v5) specifies the version of RHS
- Third segment (`oif` or `raw`) specifies whether directly or via OIF
- Fourth segment (`dopri5` or `DPjl`) specifies the used integrator:
  `dopri5` is from SciPy, `DP5jl` is from `OrdinaryDiffEq.jl`.

When we do the actual integration, we see that Julia natively is faster
25% than calling `DP5jl` via OIF from Python with RHS in Numba v4
and 33% faster in comparison with Python + SciPy's `dopri5`:
```
---
python run.py

Python native and via OIF: Scipy.integrate.ode.dopri5
# method                        1600           6400           25600
py-numba-v4-oif-dopri5          0.001 ± 0.000  0.018 ± 0.000  0.355 ± 0.001
py-numba-v4-raw-dopri5          0.001 ± 0.000  0.017 ± 0.000  0.353 ± 0.001

Julia native
method/resolution               1600           6400           25600
jl-v5-raw-DP5jl                 0.001 ± 0.000  0.010 ± 0.000  0.238 ± 0.000

Python via OIF call to `jl_diffeq` (Julia OrdinaryDiffEq.jl)
method/resolution               1600           6400           25600
py-numba-v4-oif-DP5jl           0.018 ± 0.000  0.031 ± 0.000  0.320 ± 0.000

Python native to SciPy (sanity check)
method/resolution               1600           6400           25600
py-numba-v4-raw-dopri5          0.001 ± 0.000  0.018 ± 0.000  0.313 ± 0.002
```


## Conclusion

In this experiment, we have shown that Python is still slower than Julia
during time integration even when we call `integrate` only once.

Somewhat sadly, I was not able to reproduce the results of RHS evaluation
that are in the paper: in the paper I claim that Python Numba v4
takes 0.116 seconds, but here it is 0.120; for Julia was 0.122, here it is
0.134.
