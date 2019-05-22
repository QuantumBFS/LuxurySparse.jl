export SDPermMatrix,
    SPermMatrix,
    SDSparseMatrixCSC,
    SDMatrix,
    SDDiagonal,
    SDVector,
    staticize,
    dynamicize

######### Union of static and dynamic matrices ##########
const SDMatrix{T} = Union{Matrix{T}, SArray{Shape,T,2,L} where {Shape, L}}
const SDDiagonal{T} = Union{Diagonal{T}, SDiagonal{N, T} where N}
const SDVector{T} = Union{Vector{T}, SVector{N, T} where N}
const SDPermMatrix{Tv, Ti} = PermMatrix{Tv, Ti, <: SDVector{Tv}, <: SDVector{Ti}}
const SPermMatrix{N, Tv, Ti} = PermMatrix{Tv, Ti, <:SVector{N, Tv}, <:SVector{N, Ti}}
const SDSparseMatrixCSC{Tv, Ti} = Union{SparseMatrixCSC{Tv, Ti}, SSparseMatrixCSC{Tv, Ti}}

######### staticize ##########
"""
    staticize(A::AbstractMatrix) -> AbastractMatrix

transform a matrix to a static form.
"""
function staticize end

staticize(A::AbstractMatrix) = SMatrix{size(A,1), size(A,2)}(A)
staticize(A::AbstractVector) = SVector{length(A)}(A)
staticize(A::Diagonal) = SDiagonal{size(A,1)}((A.diag...,))
staticize(A::PermMatrix) = PermMatrix(SVector{size(A,1)}(A.perm), SVector{size(A, 1)}(A.vals))
function staticize(A::SparseMatrixCSC)
    iszero(A) && return SSparseMatrixCSC(A.m, A.n, SVector{length(A.colptr)}(A.colptr), SVector{0, eltype(A.rowval)}(), SVector{0, eltype(A.nzval)}())
    SSparseMatrixCSC(A.m, A.n, SVector{length(A.colptr)}(A.colptr), SVector{length(A.rowval)}(A.rowval), SVector{length(A.nzval)}(A.nzval))
end

"""
    dynamicize(A::AbstractMatrix) -> AbastractMatrix

transform a matrix to a dynamic form.
"""
function dynamicize end

dynamicize(A::SMatrix) = Matrix(A)
dynamicize(A::SVector) = Vector(A)
dynamicize(A::SDiagonal) = Diagonal(Vector(A.diag))
dynamicize(A::PermMatrix) = PermMatrix(Vector(A.perm), Vector(A.vals))
function dynamicize(A::SSparseMatrixCSC)
    SparseMatrixCSC(A.m, A.n, Vector(A.colptr), Vector(A.rowval), Vector(A.nzval))
end
