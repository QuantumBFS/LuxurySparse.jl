export SparseMatrixCOO

"""
    SparseMatrixCOO(is::Vector, js::Vector, vs::Vector, m::Int, n::Int) -> SparseMatrixCOO
    SparseMatrixCOO{Tv, Ti}(is::Vector{Ti}, js::Vector{Ti}, vs::Vector{Tv}, m::Int, n::Int) -> SparseMatrixCOO

A sparse matrix in COOrdinate format.

Also known as the ‘ijv’ or ‘triplet’ format.

# Notes

COO matrices should not be used in arithmetic operations like addition, subtraction, multiplication, division, and matrix power.

### Advantages of the COO format
* facilitates fast conversion among sparse formats
* permits duplicate entries (see example)
* very fast conversion to and from CSR/CSC formats (CSR is not implemented)

### Disadvantages of the COO format
does not directly support:
* arithmetic operations
* slicing

### Intended Usage
* COO is a fast format for constructing sparse matrices
* Once a matrix has been constructed, convert to CSR or CSC format for fast arithmetic and matrix vector operations
* By default when converting to CSR or CSC format, duplicate (i,j) entries will be summed together. This facilitates efficient construction of finite element matrices and the like. (see example)
"""
mutable struct SparseMatrixCOO{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
    is::Vector{Ti}
    js::Vector{Ti}
    vs::Vector{Tv}
    m::Int
    n::Int

    function SparseMatrixCOO{Tv,Ti}(
        is::Vector{Ti},
        js::Vector{Ti},
        vs::Vector{Tv},
        m::Int,
        n::Int,
    ) where {Ti,Tv}
        length(is) == length(js) == length(vs) ||
            throw(ArgumentError("Input row, col, data should be equal size."))
        new{Tv,Ti}(is, js, vs, m, n)
    end
end

SparseMatrixCOO(is::Vector{Ti}, js::Vector{Ti}, vs::Vector{Tv}, m, n) where {Ti,Tv} =
    SparseMatrixCOO{Tv,Ti}(is, js, vs, m, n)

copy(coo::SparseMatrixCOO{Tv,Ti}) where {Tv,Ti} =
    SparseMatrixCOO{Tv,Ti}(copy(coo.is), copy(coo.js), copy(coo.vs), coo.m, coo.n)
function copyto!(A::SparseMatrixCOO{Tv,Ti}, B::SparseMatrixCOO{Tv,Ti}) where {Tv,Ti}
    size(A) == size(B) && nnz(A) == nnz(B) ||
        throw(MethodError("size/nnz of two coo matrices do not match!"))
    copyto!(A.is, B.is)
    copyto!(A.js, B.js)
    copyto!(A.vs, B.vs)
    A
end

function SparseMatrixCOO{Tv,Ti}(
    ::UndefInitializer,
    m::Int,
    n::Int,
    nnz::Int = 0,
) where {Tv,Ti<:Integer}
    is = Vector{Ti}(undef, nnz)
    js = Vector{Ti}(undef, nnz)
    vs = Vector{Tv}(undef, nnz)
    return SparseMatrixCOO(is, js, vs, m, n)
end

function SparseMatrixCOO{T}(::UndefInitializer, m::Int, n::Int, nnz::Int = 0) where {T}
    return SparseMatrixCOO{T,Int}(undef, m, n, nnz)
end

"""
    allocated_coo(::Type, M::Int, N::Int, nnz::Int) -> SparseMatrixCOO

Construct a preallocated `SparseMatrixCOO` instance.
"""
function allocated_coo(::Type{T}, M::Int, N::Int, nnz::Int) where {T}
    SparseMatrixCOO{T}(undef, M, N, nnz)
end

function getindex(coo::SparseMatrixCOO{Tv,Ti}, i::Integer, j::Integer) where {Tv,Ti}
    res = zero(Tv)
    for k = 1:nnz(coo)
        if coo.is[k] == i && coo.js[k] == j
            res += coo.vs[k]
        end
    end
    res
end

size(coo::SparseMatrixCOO) = (coo.m, coo.n)
size(coo::SparseMatrixCOO, axis::Int) =
    axis == 1 ? coo.m : (axis == 2 ? coo.n : throw(MethodError("invalid axis parameter")))

# SparseArrays: SparseMatrixCSC, nnz, nonzeros, dropzeros!, findnz
nnz(coo::SparseMatrixCOO) = coo.is |> length
nonzeros(coo::SparseMatrixCOO) = coo.vs

function dropzeros!(coo::SparseMatrixCOO{Tv,Ti}; trim::Bool = false) where {Tv,Ti}
    mask = abs.(coo.vs) .> 1e-15
    SparseMatrixCOO{Tv,Ti}(coo.is[mask], coo.js[mask], coo.vs[mask], coo.m, coo.n)
end

findnz(coo::SparseMatrixCOO) = (coo.is, coo.js, coo.vs)
isdense(::SparseMatrixCOO) = false

Base.@propagate_inbounds function Base.setindex!(
    coo::SparseMatrixCOO{Tv,Ti},
    v,
    i::Integer,
    j::Integer,
) where {Tv,Ti}
    @boundscheck (1 <= i <= coo.m) && (1 <= j <= coo.n) || throw(BoundsError(coo, (i, j)))

    push!(coo.is, i)
    push!(coo.js, j)
    push!(coo.vs, convert(Tv, v))
    return coo
end
