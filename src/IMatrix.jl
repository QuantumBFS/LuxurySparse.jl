"""
    IMatrix{N, Tv}()
    IMatrix{N}() where N = IMatrix{N, Int64}()
    IMatrix(A::AbstractMatrix{T}) where T -> IMatrix

IMatrix matrix, with size N as label, use `Int64` as its default type, both `*` and `kron` are optimized.
"""
struct IMatrix{N, Tv} <: AbstractLuxurySparseMatrix{Tv, Int} end
IMatrix{N}() where N = IMatrix{N, Bool}()
IMatrix(N::Int) = IMatrix{N}()

size(A::IMatrix{N}, i::Int) where N = N
size(A::IMatrix{N}) where N = (N, N)
getindex(A::IMatrix{N, T}, i::Integer, j::Integer) where {N, T} = T(i==j)

####### sparse matrix ######
nnz(M::IMatrix{N}) where N = N
nonzeros(M::IMatrix{N, T}) where {N, T} = ones(T, N)

## traits
ishermitian(::Type{<:IMatrix}) = true
ishermitian(::IMatrix) = true

issymmetric(::Type{<:IMatrix}) = true
issymmetric(::IMatrix) = true

transpose(A::IMatrix) = A

## move
similar(::IMatrix{N, Tv}, ::Type{T}) where {N, Tv, T} = IMatrix{N, T}()
copyto!(A::IMatrix{N}, ::IMatrix{N}) where N = A
