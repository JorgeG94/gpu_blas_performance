# calum_performance_tool
Tool to test dgemm performance using hipblas

# How to build 

`hipcc -L/opt/rocm-X.X.Y/lib -lhipblas performance.cpp`

# How to execute 

`./a.out 36000 14400 36000 10 N N` 

The N N represents the type of blas operation to do in the input matrices, so a comprehensive test would be:

```
./a.out 36000 14400 36000 10 N N
./a.out 36000 14400 36000 10 N T
./a.out 36000 14400 36000 10 T N 
./a.out 36000 14400 36000 10 T T 
```

Performance observed with different rocm versions: 

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
