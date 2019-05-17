################## To IMatrix ######################
IMatrix{N, T}(::AbstractMatrix) where {N, T} = IMatrix{N, T}()
IMatrix{N}(A::AbstractMatrix{T}) where {N, T} = IMatrix{N, T}()
IMatrix(A::AbstractMatrix{T}) where T = IMatrix{size(A, 1) == size(A,2) ? size(A, 2) : throw(DimensionMismatch()), T}()

################## To Diagonal ######################
Diagonal(A::PermMatrix) = Diagonal(diag(A))
Diagonal(A::IMatrix) = Eye(size(A, 1))
Diagonal{T}(::IMatrix{N}) where {T, N} = Diagonal{T}(ones(T, N))

################## To SparseMatrixCSC ######################
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
SparseMatrixCSC(coo::SparseMatrixCOO) = sparse(coo.is, coo.js, coo.vs, coo.m, coo.n)

################## To Dense ######################
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

function Matrix(coo::SparseMatrixCOO{T}) where T
    mat = zeros(T, coo.m, coo.n)
    for (i,j,v) in zip(coo.is, coo.js, coo.vs)
        mat[i,j] += v
    end
    mat
end

################## To PermMatrix ######################
PermMatrix{Tv, Ti}(::IMatrix{N}) where {Tv, Ti, N} = PermMatrix{Tv, Ti}(Vector{Ti}(1:N), ones(Tv, N))
PermMatrix{Tv}(X::IMatrix) where Tv = PermMatrix{Tv, Int}(X)
PermMatrix(X::IMatrix{N, T}) where {N, T} = PermMatrix{T, Int}(X)

PermMatrix{Tv, Ti}(A::PermMatrix) where {Tv, Ti} = PermMatrix(Vector{Ti}(A.perm), Vector{Tv}(A.vals))

function _findnz(A::AbstractMatrix)
    I = findall(!iszero, A)
    getindex.(I, 1), getindex.(I, 2), A[I]
end

_findnz(A::AbstractSparseArray) = findnz(A)

function PermMatrix{Tv, Ti}(A::AbstractMatrix) where {Tv, Ti}
    i, j, v = _findnz(A)
    j == collect(1:size(A, 2)) || throw(ArgumentError("This is not a PermMatrix"))
    order = invperm(i)
    PermMatrix{Tv, Ti}(Vector{Ti}(order), Vector{Tv}(v[order]))
end
PermMatrix(A::AbstractMatrix{T}) where T = PermMatrix{T, Int}(A)
PermMatrix(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = PermMatrix{Tv, Ti}(A) # inherit indice type
PermMatrix{Tv, Ti}(A::Diagonal{Tv}) where {Tv, Ti} = PermMatrix(Vector{Ti}(1:size(A, 1)), A.diag)
#PermMatrix(A::Diagonal{T}) where T = PermMatrix{T, Int}(A)
# lazy implementation
function PermMatrix{Tv, Ti, Vv, Vi}(A::AbstractMatrix) where {Tv, Ti<:Integer, Vv<:AbstractVector{Tv}, Vi<:AbstractVector{Ti}}
    pm = PermMatrix(PermMatrix{Tv, Ti}(A))
    PermMatrix(Vi(pm.perm), Vv(pm.vals))
end

############## To SparseMatrixCOO ##############
function SparseMatrixCOO(A::Matrix{Tv}; atol=1e-12) where Tv
    m, n = size(A)
    is = Int[]
    js = Int[]
    vs = Tv[]
    for j = 1:n
        for i = 1:m
            if abs(A[i,j]) > atol
                push!(is, i)
                push!(js, j)
                push!(vs, A[i,j])
            end
        end
    end
    SparseMatrixCOO(is, js, vs, m, n)
end

import Base: convert
convert(T::Type{<:PermMatrix}, m::AbstractMatrix) = m isa T ? m : T(m)
convert(T::Type{<:IMatrix}, m::AbstractMatrix) = m isa T ? m : T(m)
convert(T::Type{<:Diagonal}, m::AbstractMatrix) = m isa T ? m : T(m)
