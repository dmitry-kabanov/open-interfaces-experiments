# 15 Performance comparison of C and Julia, REPEAT, new resolutions

This is basically a rerun of the Exp. 12, but the resolutions are different.

## Results

New runtimes are the following:

| Method/N |      1600 |      6400 |     25600 |
| :------ | ----------- | ------- | ----- |
| C + OIF | 0.120 ± 0.003 | 1.363 ± 0.003 | 28.399 ± 0.041 |
| Julia native | 0.093 ± 0.002 | 1.344 ± 0.003 | 28.373 ± 0.029 |


We can see from the table that at $N=6400$ the difference in runtime is about
1%, while at $N=25600$ it is virtually zero.


## Conclusions

We demonstrated in this experiment that invoking Julia solvers from C using
Open Interfaces does not induce a significant performance penalty comparing
to using Julia solver directly from Julia to solve the same problem.
