"""
    LuxurySparse

A luxury sparse matrix library for Julia.

NOTE: this library may be a pirate for the following builtin array types:

- `Diagonal`
"""
module LuxurySparse

using Random, LinearAlgebra, SparseArrays

# types that we will inherit from
import SparseArrays: AbstractSparseArray, AbstractSparseMatrix

# types will be overloaded by some pirate methods
import LinearAlgebra: Diagonal
import SparseArrays: SparseMatrixCSC

# APIs that we will overload for
## standard array interface
import Base: getindex, size, similar, copy, show, copyto!, inv, mul!

## linear algebra
import LinearAlgebra: ishermitian, issymmetric, diag, logdet
export ishermitian, issymmetric

## sparse arrays
import SparseArrays: nnz, nonzeros, dropzeros!, issparse
export nnz, nonzeros, dropzeros!, issparse

################################################################################

export isdense

# additional traits
"""
    isdense(T) -> Bool
    isdense(A) -> Bool

Determine whether type `T` or array `A` is dense.
"""
function isdense end

# subtype of DenseArray is dense by default
isdense(::DenseArray) = true
isdense(::Type{<:DenseArray}) = true

isdense(::Type{<:AbstractSparseArray}) = false
isdense(::AbstractSparseArray) = false

# TODO: this need to be added to upstream
isdense(::Diagonal) = false
issparse(::Diagonal) = true

abstract type AbstractLuxurySparseMatrix{Tv, Ti} <: AbstractSparseMatrix{Tv, Ti} end

# IMatrix
export IMatrix
include("IMatrix.jl")

# PermMatrix
export PermMatrix
include("PermMatrix.jl")

include("conversions.jl")

export pmrand
include("random.jl")

end # module
