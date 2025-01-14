# 14: Comparing C solvers using an implementation via OIF or directly

## Goal

To demonstrate that there is no penalty in using Open Interfaces, at least
in the cases where a user's code and an implementation are in the same
language.


## Procedure

We solve the Burgers' equation using a Dopri5 method that I have reimplemented
from looking at the Fortran original code (partially implemented as I did
not consider setting options or dense output).


## Results

I first was compiling both solvers OIF and RAW using the Clang C compiler.
However, the Open Interfaces themselves were compiled using GCC as somehow
building with Clang does not work.

Here I got slightly weird results where OIF version is /faster/ than the RAW
version:
```
# method                        1600         6400         25600
oif                             0.09 ± 0.00  1.51 ± 0.01  25.88 ± 0.05
raw                             0.10 ± 0.00  1.58 ± 0.01  26.99 ± 0.06
```
which is weird as the RAW version is compiled completely (including Dopri5)
using Clang, and the Clang usually gives faster code.

When I have excluded any discrepancies in codes or solutions, then only
possibility left was to switch to using GCC for everything.
These are the results:
```
# method                        1600         6400         25600
oif                             0.12 ± 0.00  1.90 ± 0.01  31.74 ± 0.06
raw                             0.12 ± 0.00  1.84 ± 0.01  31.58 ± 0.05
```

We can see that for the grid $N = 25600$, the performance of OIF version
is only 0,5% slower than the RAW version.
Overall, these results are consistent and "normal", that is,
it is expected that the OIF version is slightly slower than the RAW version.


## Conclusions

In this experiment, we demonstrate that using Open Interfaces from C
with a C implementation does not bring any significant performance penalties
in comparison with using the implementation directly.
