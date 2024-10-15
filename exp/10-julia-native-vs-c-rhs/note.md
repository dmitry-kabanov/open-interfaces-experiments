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


### Line profiling Python wrapper for C function

Results are:
```
➜ kernprof -l -v call_ivp_python.py
Comparing performance of Open Interfaces for IVP interface from Python
BEGIN warmup
END warmup

Resolution N = 3200
Measure performance 2 times
Numpy
Numpy
Runtime, sec: 3.196 ± 0.001
ELAPSED_TIME mean: 2.125
Wrote profile results to call_ivp_python.py.lprof
Timer unit: 1e-06 s

Total time: 2.44674 s
File: call_ivp_python.py
Function: compute_rhs_wrapper at line 47

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    47                                               @profile
    48                                               def compute_rhs_wrapper(udot, u, p, t):
    49                                                   # Load C function
    50                                                   # Call it with arguments (t, u, udot, p)
    51                                                   global ELAPSED_TIME
    52     41290      26294.0      0.6      1.1          tic = time.perf_counter()
    53     41290      26366.1      0.6      1.1          if isinstance(u, VectorValue):
    54     41289     284583.2      6.9     11.6              np_u = u.to_numpy(dtype=np.float64, copy=False)
    55     41289     149975.3      3.6      6.1              np_udot = udot.to_numpy(dtype=np.float64, copy=False)
    56                                                       # np_udot = np.asarray(udot)
    57                                                   else:
    58         1         19.1     19.1      0.0              print("Numpy")
    59                                                       # sys.exit()
    60         1          0.2      0.2      0.0              np_u = u
    61         1          0.2      0.2      0.0              np_udot = udot
    62                                                   # c_u = np_u.ctypes.data_as(double_p_t)
    63                                                   # c_udot = np_udot.ctypes.data_as(double_p_t)
    64                                                   # c_u = np_u.ctypes._as_parameter_
    65                                                   # c_udot = np_udot.ctypes.data_as(double_p_t)
    66     41290     808564.5     19.6     33.0          c_u = ctypes.cast(np.ctypeslib.as_ctypes(np_u), double_p_t)
    67     41290     510567.6     12.4     20.9          c_udot = ctypes.cast(np.ctypeslib.as_ctypes(np_udot), double_p_t)
    68                                                   # c_u = ctypes.cast(np_u.__array_interface__["data"][0], double_p_t)
    69                                                   # c_udot = np_udot.__array_interface__["data"][0], double_p_t)
    70                                                   # c_u = np_u.ctypes.data_as(ctypes.c_void_p)
    71                                                   # c_udot = np_udot.ctypes.data_as(ctypes.c_void_p)
    72                                                   # c_u = np_u.ctypes.data
    73                                                   # c_udot = np_udot.ctypes.data
    74                                                   # c_u = ctypes.cast(memoryview(np_u)[:], double_p_t)
    75                                                   # c_udot = ctypes.cast(memoryview(np_udot)[:], double_p_t)
    76     41290      69291.6      1.7      2.8          x = ctypes.pointer(ctypes.c_double(p[0]))
    77     41290      20971.4      0.5      0.9          toc = time.perf_counter()
    78     41290      24984.5      0.6      1.0          ELAPSED_TIME += toc - tic
    79     41290     525125.0     12.7     21.5          compute_rhs(t, c_u, c_udot, x, len(u))
    80                                                   # compute_rhs(t, np_u, np_udot, x, len(u))
```

It seems that the translation from C arrays to NumPy arrays take an
extraordinary amount of runtime.


## Conclusion

The main obstacle to getting good performance right now is that somehow
translation of arrays between languages takes large portion of runtime---
simple operation like getting a pointer to a data buffer is comparable
to computations of the right-hand side.
