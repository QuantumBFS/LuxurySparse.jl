# LuxurySparse.jl

[![Build Status](https://github.com/QuantumBFS/LuxurySparse.jl/workflows/CI/badge.svg)](https://github.com/QuantumBFS/LuxurySparse.jl/actions)
[![Codecov](https://codecov.io/gh/QuantumBFS/LuxurySparse.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/QuantumBFS/LuxurySparse.jl)

High performance extension for sparse matrices.

## Contents
* General Permutation Matrix `PermMatrix`,
* Identity Matrix `IMatrix`,
* Coordinate Format Matrix `SparseMatrixCOO`,
* Static Matrices `SSparseMatrixCSC`, `SPermMatrix` et. al.

with high performance type conversion, `kron`, and multiplication operations.

## Installation
Install with the package manager, `pkg> add LuxurySparse`.

## Usage

```julia
using SparseArrays, LuxurySparse, BenchmarkTools

pm = pmrand(7)  # a random permutation matrix
id = IMatrix(3) # an identity matrix
@btime fastkron(pm, id)   # 96.606 ns

Spm = pm |> SparseMatrixCSC  # convert to SparseMatrixCSC
Sid = id |> SparseMatrixCSC
@btime kron(Spm, Sid)     # 388.817 ns

spm = pm |> staticize        # convert to static matrix, notice that `id` is already static.
@btime fastkron(spm, spm)    # 104.581 ns
@btime fastkron(pm, pm)      # 146.484 ns
```

For more information, please refer the latest [Documentation](https://quantumbfs.github.io/LuxurySparse.jl/latest/).

## Planned features
* Change `PermMatrix` to column major
* Better support of conversion to static matrices
