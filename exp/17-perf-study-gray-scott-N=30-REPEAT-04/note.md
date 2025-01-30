# 17 Rerun of Gray--Scott performance study

## Goal

Rerun a performance comparison that I did in September 2024.
A lot of things changed since that time, so I want to make sure
that the results are still valid.


## Procedure

We use SUNDIALS CVODE Adams-Moulton method to time-integrate 2D Gray--Scott
system.
We integrate via Open Interfaces or via Cython bindings scikit.odes.

We run each simulation 30 times and report 95% confidence intervals.


## Results

```
           N                  64                128               256               512
sundials_cvode            0.114   0.01      0.386   0.00      1.435   0.01     11.396   0.07
native_sundials_cvode     0.120   0.06      0.385   0.00      1.487   0.01     11.956   0.07
```


## Conclusions

We compared performance of time integration using SUNDIALS CVODE via Open
Interfaces and via Cython bindings scikit.odes.
Somehow, the performance of the Cython bindings is slightly worse than the
performance of the Open Interfaces.
