# Tutorial

## Generalized Permutation Matrix

### Example: Control-Y Gate in Quantum simulation
[Generalized permutation matrices](https://en.wikipedia.org/wiki/Generalized_permutation_matrix) are frequently used in fields such as quantum computation and group theory. Here we see an example of Control-Y Gate in matrix form:
```math
\left(\begin{matrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & -i\\
0 & 0 & i & 0\\
\end{matrix}\right)
```

We can represent the matrix using a `PermMatrix` type with the following fields:

* `perm`: [1, 2, 4, 3]
* `vals`: [1, 1, -i, i]

Now let's do a benchmark to feel the speed up

```@example permmatrix
using LuxurySparse: PermMatrix
pm = PermMatrix([1,2,4,3], [1,1,-im,im])
```

```julia
using BenchmarkTools
v = randn(4)
@benchmark $pm*$v samples=100000 evals=1000
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
@benchmark $sp*$v samples=100000 evals=1000
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
The identity matrix is static and is defined as:
```
struct IMatrix{Tv} <: AbstractMatrix{Tv} end
```

With this type, the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) operation can be much faster. Now let's see a benchmark

```@example identity
using LuxurySparse: IMatrix
Id = IMatrix{Float64}(1)
B = randn(7,7);
```

```julia
using BenchmarkTools
@benchmark kron($Id, $B) samples=100000 evals=1000
```
```
BenchmarkTools.Trial: 100000 samples with 1000 evaluations.
 Range (min … max):  3.333 ns … 22.667 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.417 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.457 ns ±  0.343 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

        ▄     █      ▆     ▄      ▅     ▄      ▃     ▃       ▃
  ▆▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▇ █
  3.33 ns      Histogram: log(frequency) by time     3.71 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```
As a comparison
```julia
using LinearAlgebra, SparseArrays
spI = sparse(I, 7, 7)
@benchmark kron($spI, $B) samples=100000 evals=1000 seconds=3600
```
```
julia> @benchmark kron($spI, $B) samples=100000 evals=1000 seconds=3600
BenchmarkTools.Trial: 100000 samples with 1000 evaluations.
 Range (min … max):  694.417 ns …  23.188 μs  ┊ GC (min … max):  0.00% …  0.00%
 Time  (median):       1.117 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):     1.499 μs ± 977.508 ns  ┊ GC (mean ± σ):  28.57% ± 26.68%

       ▃██▃                                                      
  ▃▃▃▄▅████▆▄▃▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▃▃▃▃▃▃▃▃▃▃▃▃▂▂▂▂ ▃
  694 ns           Histogram: frequency by time         4.23 μs <

 Memory estimate: 7.11 KiB, allocs estimate: 8.

```

With the help of Julia's multiple dispatch, a more performant `kron` operation can be implemented with LuxurySparse's `IMatrix` type versus the standard sparse identity matrix.
