# calum_performance_tool
Tool to test dgemm performance using hipblas

# How to build 

```
git clone git@github.com:JorgeG94/calum_performance_tool.git
mkdir build 
cd build
cmake ../
make 
```

# How to execute 

`./performance 36000 14400 36000 10 N N 10` 

The N N represents the type of blas operation to do in the input matrices, so a comprehensive test would be:

```
./a.out 36000 14400 36000 10 N N 10 
./a.out 36000 14400 36000 10 N T 10 
./a.out 36000 14400 36000 10 T N 10
./a.out 36000 14400 36000 10 T T 10
```

