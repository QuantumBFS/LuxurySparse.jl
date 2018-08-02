# IMatrix
## TODO: document this
IMatrix{N, T}(::AbstractMatrix) where {N, T} = IMatrix{N, T}()
IMatrix{N}(A::AbstractMatrix{T}) where {N, T} = IMatrix{N, T}()
IMatrix(A::AbstractMatrix{T}) where T = IMatrix{size(A, 1) == size(A,2) ? size(A, 2) : throw(DimensionMismatch()), T}()



# PermMatrix
## TODO: document this

## forward static parameter for sparse arrays
PermMatrix(A::AbstractSparseMatrix{Tv, Ti}) where {Tv, Ti} = PermMatrix{Tv, Ti}(A)

PermMatrix{Tv, Ti}(::IMatrix{N}) where {Tv, Ti, N} = PermMatrix{Tv, Ti}(Vector{Ti}(1:N), ones(Tv, N))
PermMatrix{Tv, Ti}(A::PermMatrix) where {Tv, Ti} = PermMatrix(Vector{Ti}(A.perm), Vector{Tv}(A.vals))

## default indice type is Int
PermMatrix(A::AbstractMatrix{T}) where T = PermMatrix{T, Int}(A)
PermMatrix{Tv, Ti}(A::Diagonal) where {Tv, Ti} = PermMatrix{Tv, Ti}(Vector{Ti}(1:size(A, 1)), Vector{Tv}(A.diag))

## check if this is a PermMatrix for other matrix
function PermMatrix{Tv, Ti, Vv, Vi}(A::AbstractMatrix) where {Tv, Ti, Vv, Vi}
    i,j,v = findnz(A)
    j == collect(1:size(A, 2)) || throw(InexactError(:PermMatrix, PermMatrix, A))
    order = invperm(i)
    PermMatrix{Tv, Ti}(Vi(order), Vv(v[order]))
end

PermMatrix{Tv, Ti}(A::AbstractMatrix) where {Tv, Ti} = PermMatrix{Tv, Ti, Vector{Tv}, Vector{Ti}}(A)



# Matrix
Matrix{T}(::IMatrix{N}) where {T, N} = Matrix{T}(I, N, N)
Matrix(::IMatrix{N, T}) where {N, T} = Matrix{T}(I, N, N)

function Matrix{T}(X::PermMatrix) where T
    n = size(X, 1)
    Mf = zeros(T, n, n)
    @simd for i=1:n
        @inbounds Mf[i, X.perm[i]] = X.vals[i]
    end
    return Mf
end
Matrix(X::PermMatrix{T}) where T = Matrix{T}(X)



# SparseMatrixCSC
SparseMatrixCSC{Tv, Ti}(A::IMatrix{N}) where {Tv, Ti <: Integer, N} = SparseMatrixCSC{Tv, Ti}(I, N, N)
SparseMatrixCSC{Tv}(A::IMatrix) where Tv = SparseMatrixCSC{Tv, Int}(A)
SparseMatrixCSC(A::IMatrix{N, T}) where {N, T} = SparseMatrixCSC{T, Int}(I, N, N)
function SparseMatrixCSC(M::PermMatrix)
    n = size(M, 1)
    #SparseMatrixCSC(n, n, collect(1:n+1), M.perm, M.vals)
    order = invperm(M.perm)
    SparseMatrixCSC(n, n, collect(1:n+1), order, M.vals[order])
end
SparseMatrixCSC{Tv, Ti}(M::PermMatrix{Tv, Ti}) where {Tv, Ti} = SparseMatrixCSC(M)

function SparseMatrixCSC(M::Diagonal)
    n = size(M, 1)
    SparseMatrixCSC(n, n, collect(1:n+1), collect(1:n), M.diag)
end
SparseMatrixCSC{Tv, Ti}(M::Diagonal{Tv}) where {Tv, Ti} = SparseMatrixCSC(M)
