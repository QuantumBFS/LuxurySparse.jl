module LuxurySparse

using LinearAlgebra, SparseArrays, Random
using StaticArrays: SVector, SMatrix, SDiagonal, SArray
using SparseArrays: SparseMatrixCSC
using SparseArrays.HigherOrderFns
using Base: @propagate_inbounds
using LinearAlgebra
using LinearAlgebra: StructuredMatrixStyle
using Base.Broadcast:
    BroadcastStyle, AbstractArrayStyle, Broadcasted, DefaultArrayStyle, materialize!

# static types
export SDPermMatrix, SPermMatrix, PermMatrix, pmrand,
    SDSparseMatrixCSC, SSparseMatrixCSC, SparseMatrixCSC, sprand,
    SparseMatrixCOO,
    SDMatrix, SDVector,
    SDDiagonal, Diagonal,
    IMatrix,
    staticize, dynamicize,
    fast_invperm,
    IterNz

include("utils.jl")
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

include("iterate.jl")

end
