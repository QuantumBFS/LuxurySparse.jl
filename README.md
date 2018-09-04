# LuxurySparse.jl

[![Build Status](https://travis-ci.org/QuantumBFS/LuxurySparse.jl.svg?branch=master)](https://travis-ci.org/QuantumBFS/LuxurySparse.jl)

High performance extension for sparse matrices.

## Contents
* General Permutation Matrix `PermMatrix`,
* Identity Matrix `IMatrix`,
* Coordinate Format Matrix `SparseMatrixCOO`,
* Static Matrices `SSparseMatrixCSC`, `SPermMatrix` et. al.

with high performance `type convertion`, `kron` and `multiplication` operations.

## Installation
Install with the package manager, `pkg> add LuxurySparse`.

## How to use
Here is a simple example

```julia
using SparseArrays
using LuxurySparse
using BenchmarkTools

pm = pmrand(7)  # a random permutation matrix
id = IMatrix(3) # an identity matrix
@benchmark kron(pm, id) # kronecker product

Spm = pm |> SparseMatrixCSC  # convertion to SparseMatrixCSC
Sid = id |> SparseMatrixCSC
@benchmark kron(Spm, Sid)    # compare the performance.

spm = pm |> staticize        # convertion to static matrices, notice `id` is already static.
@benchmark kron(spm, spm)    # compare the performance.
@benchmark kron(pm, pm)    # compare the performance.
```

For more information, please refer the latest [document](https://quantumbfs.github.io/LuxurySparse.jl/latest/).

## Planned features
* Change `PermMatrix` to column major
* Better support to static matrices.
