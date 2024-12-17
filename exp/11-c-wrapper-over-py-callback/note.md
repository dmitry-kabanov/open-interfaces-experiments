# 11 Python directly works with C and Julia

## Description

In this experiment I use a C RHS function, load it in Python, and then invoke
`OrdinaryDiffEq.jl` directly from Python.

The goal is to see how much slower using `OrdinaryDiffEq.jl` from Python is
relative to just running in Julia.


## Procedure

I use package JuliaCall https://juliapy.github.io/PythonCall.jl/stable/juliacall/
to call Julia from Python.
I first used package `PyJulia` but it seems to be slow (solving ODE at N=3200
takes about 1-1.5 seconds; probably arrays are copied).
Also the package `JuliaCall` is recommended.

This package has a peculiarity that it creates a new Julia environment.
Therefore, one needs to install required packages into this environment.
The thing that worked for me is to create `juliapkg.json` file and then
during the import it just adds and precompiles the packages.


### Statistical results

I run simulations 30 times for several different resolutions.
The simulation are run from Julia directly and from Python directly using
`OrdinaryDiffEq.jl` package (without OIF).

These are the results:
```
method/resolution               200            400            800            1600           3200           6400
jl-native-v5                    0.00 ± 0.00    0.01 ± 0.00    0.02 ± 0.00    0.09 ± 0.00    0.34 ± 0.00    1.53 ± 0.01
OrdinaryDiffEq.jl from Python   0.11 ± 0.01    0.20 ± 0.01    0.39 ± 0.01    0.808 ± 0.009  1.80 ± 0.01    4.80 ± 0.10
```


### Line profiling Python wrapper for C function

Results are:
```
➜ kernprof -l -v call_ivp_python.py --nruns 1
Comparing performance of Open Interfaces for IVP interface from Python
BEGIN warmup
END warmup

Resolution N = 3200
Measure performance 2 times
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
