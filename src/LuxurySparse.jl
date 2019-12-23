module LuxurySparse

using LinearAlgebra, SparseArrays, Random
using StaticArrays: SVector, SMatrix, SDiagonal, SArray
using Base: @propagate_inbounds

import Base: copyto!, *, kron, -
import LinearAlgebra: ishermitian
import Base: getindex, size, similar, copy, show

export I, fast_invperm, isdense, allocated_coo

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
include("broadcast.jl")

end
