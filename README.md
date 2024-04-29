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

## Use magma 

`cmake -DUSE_MAGMA=True ../`

# How to execute a DGEMM

`./dgemm 36000 14400 36000 10 N N 10` 

The N N represents the type of blas operation to do in the input matrices, so a comprehensive test would be:

```
./dgemm 36000 14400 36000 10 N N 10 
./dgemm 36000 14400 36000 10 N T 10 
./dgemm 36000 14400 36000 10 T N 10
./dgemm 36000 14400 36000 10 T T 10
```

# How to execute a DSYEVD
This will run a 10 by 10 DSYEVD 10 times, repeated 10 times.
`./dsyevd 10 10 10 10 `
