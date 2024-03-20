
# Performance observed with different rocm versions: 

```
crusher170:~/crusher_dev/marco_exess> hipcc -L/opt/rocm-5.1.0/lib -lhipblas calum_performance.cpp
crusher170:~/crusher_dev/marco_exess> ./a.out 36000 14400 36000 10 T T
Performing 10 repetitions of 36000 14400 36000
Time     134714254us
GFLOP/s: 2770.66
crusher170:~/crusher_dev/marco_exess> hipcc -L/opt/rocm-5.2.0/lib -lhipblas calum_performance.cpp
crusher170:~/crusher_dev/marco_exess> ./a.out 36000 14400 36000 10 T T
Performing 10 repetitions of 36000 14400 36000
Time     134716222us
GFLOP/s: 2770.62
crusher170:~/crusher_dev/marco_exess> hipcc -L/opt/rocm-5.3.0/lib -lhipblas calum_performance.cpp
crusher170:~/crusher_dev/marco_exess> ./a.out 36000 14400 36000 10 T T
Performing 10 repetitions of 36000 14400 36000
Time     9947263us
GFLOP/s: 37522.7
```
