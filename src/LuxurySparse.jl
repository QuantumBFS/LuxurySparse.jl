module LuxurySparse

using LinearAlgebra, SparseArrays, Random
using StaticArrays: SVector, SMatrix, SDiagonal, SArray
using Base: @propagate_inbounds

import Base: copyto!, *, kron, -
import LinearAlgebra: ishermitian
import SparseArrays: SparseMatrixCSC, nnz, nonzeros, dropzeros!, findnz, issparse
import Base: getindex, size, similar, copy, show

export PermMatrix, pmrand, IMatrix, I, fast_invperm, isdense, SparseMatrixCOO, allocated_coo
export staticize, SSparseMatrixCSC, SPermMatrix, SDPermMatrix, SDSparseMatrixCSC, dynamicize
export SDMatrix, SDDiagonal, SDVector

include("Core.jl")
include("IMatrix.jl")
include("PermMatrix.jl")
include("SparseMatrixCOO.jl")
include("SSparseMatrixCSC.jl")

include("conversions.jl")
include("promotions.jl")
include("staticize.jl")
include("arraymath.jl")
include("linalg.jl")
include("kronecker.jl")

end
