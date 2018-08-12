module LuxurySparse

using LinearAlgebra, SparseArrays, Random
using StaticArrays: SVector, SMatrix

import Base: copyto!
import LinearAlgebra: ishermitian
import SparseArrays: SparseMatrixCSC, nnz, nonzeros, dropzeros!, findnz
import Base: getindex, size, similar, copy, show

export PermMatrix, pmrand, IMatrix, I, fast_invperm, notdense, SparseMatrixCOO, allocated_coo
export staticize, SSparseMatrixCSC, SDiagonal

include("Core.jl")
include("IMatrix.jl")
include("PermMatrix.jl")
include("SparseMatrixCOO.jl")

include("conversions.jl")
include("promotions.jl")
include("staticize.jl")
include("arraymath.jl")
include("linalg.jl")
include("kronecker.jl")

end
