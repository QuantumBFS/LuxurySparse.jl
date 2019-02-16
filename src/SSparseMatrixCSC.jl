export SSparseMatrixCSC

"""
    SSparseMatrixCSC{Tv,Ti<:Integer, NNZ, NP} <: AbstractSparseMatrix{Tv,Ti}

static version of SparseMatrixCSC
"""
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

function getindex(ssp::SSparseMatrixCSC{Tv}, i, j) where Tv
    S = ssp.colptr[j]
    E = ssp.colptr[j+1]-1
    for ii in S:E
        if i == ssp.rowval[ii]
            return ssp.nzval[ii]
        end
    end
    return Tv(0)
end

issparse(::SSparseMatrixCSC) = true
nonzeros(M::SSparseMatrixCSC) = M.nzval
nnz(spm::SSparseMatrixCSC{Tv, Ti, NNZ}) where {Tv, Ti, NNZ} = NNZ
dropzeros!(M::SSparseMatrixCSC; trim::Bool=false) = M

isdense(::SSparseMatrixCSC) = false
