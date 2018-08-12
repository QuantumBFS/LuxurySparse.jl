# Tutorial

## Generalized permutation matrix

### Example: Control-Y Gate in Quantum simulation
[Generalized permutation matrices](https://en.wikipedia.org/wiki/Generalized_permutation_matrix) are frequently used in fields such as quantum computation, group thoery. Here we see an example of Control-Y Gate
```math
\left(\begin{matrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & -i\\
0 & 0 & i & 0\\
\end{matrix}\right)
```

This data structure can be represented in the form of `PermMatrix`
* perm: [1, 2, 4, 3]
* vals: [1, 1, -i, i]

Now let's do a benchmark to feel the speed up

```@example permmatrix
using LuxurySparse: PermMatrix
pm = PermMatrix([1,2,4,3], [1,1,-im,im])
```

```julia
v = randn(4)
@benchmark $pm*$v
```
```
BenchmarkTools.Trial: 
  memory estimate:  144 bytes
  allocs estimate:  1
  --------------
  minimum time:     36.789 ns (0.00% GC)
  median time:      38.816 ns (0.00% GC)
  mean time:        49.227 ns (10.20% GC)
  maximum time:     1.629 μs (89.36% GC)
  --------------
  samples:          10000
  evals/sample:     992
```

As a comparison
```julia
sp = SparseMatrixCSC(pm)
@benchmark $sp*$v
```
```
BenchmarkTools.Trial: 
  memory estimate:  144 bytes
  allocs estimate:  1
  --------------
  minimum time:     64.578 ns (0.00% GC)
  median time:      65.769 ns (0.00% GC)
  mean time:        74.292 ns (6.80% GC)
  maximum time:     1.419 μs (87.15% GC)
  --------------
  samples:          10000
  evals/sample:     979
```

## Identity Matrix
Identity matrix is static, which is defined as
```
struct IMatrix{N, Tv} <: AbstractMatrix{Tv} end
```

With this type, the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) operation can be much faster. Now let's see a benchmark

```@example identity
using LuxurySparse: IMatrix
Id = IMatrix{1, Float64}()
B = randn(7,7);
```

```julia
@benchmark kron($Id, $B)
```
```
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     1.642 ns (0.00% GC)
  median time:      1.651 ns (0.00% GC)
  mean time:        1.658 ns (0.00% GC)
  maximum time:     32.101 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1000
```
With the help of Julia's multiple dispatch, the above trivil `kron` operation can be avoided.
