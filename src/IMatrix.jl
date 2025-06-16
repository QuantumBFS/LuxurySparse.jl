"""
    IMatrix{Tv}

    IMatrix(n) -> IMatrix
    IMatrix(A::AbstractMatrix{T}) where T -> IMatrix

Represents the Identity matrix with size `n`. `Int64` is its default type. Both `*` and `kron` are optimized.

# Example

```julia-repl
julia> IMatrix(4)
4×4 IMatrix{Bool}:
 1  0  0  0
 0  1  0  0
 0  0  1  0
 0  0  0  1

julia> IMatrix(rand(4,4))
4×4 IMatrix{Float64}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0

```
"""
struct IMatrix{Tv} <: AbstractMatrix{Tv}
    n::Int
end
IMatrix(n::Integer) = IMatrix{Bool}(n)

Base.size(A::IMatrix, i::Int) = (@assert i == 1 || i == 2; A.n)
Base.size(A::IMatrix) = (A.n, A.n)
Base.getindex(::IMatrix{T}, i::Integer, j::Integer) where {T} = convert(T, i == j)

Base.:(==)(d1::IMatrix, d2::IMatrix) = d1.n == d2.n
Base.isapprox(d1::IMatrix, d2::IMatrix; kwargs...) = d1 == d2

Base.similar(A::IMatrix{Tv}, ::Type{T}) where {Tv,T} = IMatrix{T}(A.n)
function Base.copyto!(A::IMatrix, B::IMatrix)
    if A.n != B.n
        throw(DimensionMismatch("matrix dimension mismatch, got $(A.n) and $(B.n)"))
    end
    A
end
LinearAlgebra.ishermitian(D::IMatrix) = true

####### sparse matrix ######
SparseArrays.nnz(M::IMatrix) = M.n
SparseArrays.findnz(M::IMatrix{T}) where {T} = (collect(1:M.n), collect(1:M.n), ones(T, M.n))
