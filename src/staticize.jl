struct SSparseMatrixCSC{Tv,Ti<:Integer, NNZ, NP} <: AbstractSparseMatrix{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::SVector{NP,Ti}      # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::SVector{NNZ,Ti}      # Row values of nonzeros
    nzval::SVector{NNZ,Tv}       # Nonzero values

    function SSparseMatrixCSC{Tv,Ti, NNZ, NP}(m::Integer, n::Integer, colptr::SVector{NP, Ti}, rowval::SVector{NNZ,Ti},
                                    nzval::SVector{NNZ,Tv}) where {Tv,Ti<:Integer, NNZ, NP}
        m < 0 && throw(ArgumentError("number of rows (m) must be ≥ 0, got $m"))
        n < 0 && throw(ArgumentError("number of columns (n) must be ≥ 0, got $n"))
        new(Int(m), Int(n), colptr, rowval, nzval)
    end
end

function SSparseMatrixCSC(m::Integer, n::Integer, colptr::SVector, rowval::SVector, nzval::SVector)
    Tv = eltype(nzval)
    Ti = promote_type(eltype(colptr), eltype(rowval))
    SSparseMatrixCSC{Tv,Ti,length(nzval),n+1}(m, n, colptr, rowval, nzval)
end

function size(spm::SSparseMatrixCSC{Tv, Ti, NNZ, NP}, i::Integer) where {Tv, Ti, NNZ, NP}
    i == 1 ? spm.m : (i == 2 ? NP-1 : throw(ArgumentError("dimension out of bound!")))
end
size(spm::SSparseMatrixCSC{Tv, Ti, NNZ, NP}) where {Tv, Ti, NNZ, NP} = (spm.m, NP-1)
SparseArrays.nnz(spm::SSparseMatrixCSC{Tv, Ti, NNZ}) where {Tv, Ti, NNZ} = NNZ

struct SPermMatrix{N, Tv, Ti<:Integer} <: AbstractMatrix{Tv}
    perm::SVector{N, Ti}   # new orders
    vals::SVector{N, Tv}   # multiplied values.

    function SPermMatrix(perm::SVector{N, Ti}, vals::SVector{N, Tv}) where {N, Tv, Ti<:Integer}
        if length(perm) != length(vals)
            throw(DimensionMismatch("permutation ($N) and multiply ($N) length mismatch."))
        end
        new{N, Tv, Ti}(perm, vals)
    end
end

size(spm::SPermMatrix{N}, i::Integer) where N = N
size(spm::SPermMatrix{N}) where N = (N, N)

######### staticize ##########
"""
    staticize(A::AbstractMatrix) -> AbastractMatrix

transform a matrix to a static form.
"""
function staticize end

staticize(A::AbstractMatrix) = SMatrix{size(A,1), size(A,2)}(A)
staticize(A::AbstractVector) = SVector{length(A)}(A)
staticize(A::Diagonal) = SDiagonal{size(A,1)}(A.diag)
staticize(A::PermMatrix) = SPermMatrix(SVector{size(A,1)}(A.perm), SVector{size(A, 1)}(A.vals))
function staticize(A::SparseMatrixCSC)
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
dynamicize(A::SPermMatrix) = PermMatrix(Vector(A.perm), Vector(A.vals))
function dynamicize(A::SSparseMatrixCSC)
    SparseMatrixCSC(A.m, A.n, Vector(A.colptr), Vector(A.rowval), Vector(A.nzval))
end


######### Union of static and dynamic matrices ##########
const SDPermMatrix = Union{PermMatrix, SPermMatrix}
const SDSparseMatrixCSC = Union{SparseMatrixCSC, SSparseMatrixCSC}
const SDMatrix = Union{Matrix, SMatrix}
const SDDiagonal = Union{Diagonal, SDiagonal}
const SDVector = Union{Vector, SVector}
