export IMatrix

"""
    IMatrix{Tv}

    IMatrix(n) -> IMatrix
    IMatrix(A::AbstractMatrix{T}) where T -> IMatrix

IMatrix matrix, with size N as label, use `Int64` as its default type, both `*` and `kron` are optimized.
"""
struct IMatrix{Tv} <: AbstractMatrix{Tv}
    n::Int
end
IMatrix(n::Integer) = IMatrix{Bool}(n)

size(A::IMatrix, i::Int) = A.n
size(A::IMatrix) = (A.n, A.n)
getindex(::IMatrix{T}, i::Integer, j::Integer) where {T} = T(i == j)

Base.:(==)(d1::IMatrix, d2::IMatrix) = d1.n == d2.n
Base.isapprox(d1::IMatrix, d2::IMatrix; kwargs...) = d1 == d2

####### sparse matrix ######
nnz(M::IMatrix) = M.n
nonzeros(M::IMatrix{T}) where {T} = ones(T, M.n)
findnz(M::IMatrix{T}) where {T} = (collect(1:M.n), collect(1:M.n), ones(T, M.n))
ishermitian(D::IMatrix) = true
isdense(::IMatrix) = false

similar(A::IMatrix{Tv}, ::Type{T}) where {Tv,T} = IMatrix{T}(A.n)
function copyto!(A::IMatrix, B::IMatrix)
    if A.n != B.n
        DimensionMismatch("matrix dimension mismatch, got $(A.n) and $(B.n)")
    end
    A
end