export SSparseMatrixCSC

@static if VERSION < v"1.4.0"

    """
        SSparseMatrixCSC{Tv,Ti<:Integer, NNZ, NP} <: AbstractSparseMatrix{Tv,Ti}

    static version of SparseMatrixCSC
    """
    struct SSparseMatrixCSC{Tv,Ti<:Integer,NNZ,NP} <: AbstractSparseMatrix{Tv,Ti}
        m::Int                  # Number of rows
        n::Int                  # Number of columns
        colptr::SVector{NP,Ti}      # Column i is in colptr[i]:(colptr[i+1]-1)
        rowval::SVector{NNZ,Ti}      # Row values of nonzeros
        nzval::SVector{NNZ,Tv}       # Nonzero values

        function SSparseMatrixCSC{Tv,Ti,NNZ,NP}(
            m::Integer,
            n::Integer,
            colptr::SVector{NP,Ti},
            rowval::SVector{NNZ,Ti},
            nzval::SVector{NNZ,Tv},
        ) where {Tv,Ti<:Integer,NNZ,NP}
            m < 0 && throw(ArgumentError("number of rows (m) must be ≥ 0, got $m"))
            n < 0 && throw(ArgumentError("number of columns (n) must be ≥ 0, got $n"))
            new(Int(m), Int(n), colptr, rowval, nzval)
        end
    end

else
    # NOTE: from 1.4.0, by subtyping AbstractSparseMatrixCSC, things like sparse broadcast
    # should just work.

    """
        SSparseMatrixCSC{Tv,Ti<:Integer, NNZ, NP} <: AbstractSparseMatrix{Tv,Ti}

    static version of SparseMatrixCSC
    """
    struct SSparseMatrixCSC{Tv,Ti<:Integer,NNZ,NP} <:
           SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}
        m::Int                  # Number of rows
        n::Int                  # Number of columns
        colptr::SVector{NP,Ti}      # Column i is in colptr[i]:(colptr[i+1]-1)
        rowval::SVector{NNZ,Ti}      # Row values of nonzeros
        nzval::SVector{NNZ,Tv}       # Nonzero values

        function SSparseMatrixCSC{Tv,Ti,NNZ,NP}(
            m::Integer,
            n::Integer,
            colptr::SVector{NP,Ti},
            rowval::SVector{NNZ,Ti},
            nzval::SVector{NNZ,Tv},
        ) where {Tv,Ti<:Integer,NNZ,NP}
            m < 0 && throw(ArgumentError("number of rows (m) must be ≥ 0, got $m"))
            n < 0 && throw(ArgumentError("number of columns (n) must be ≥ 0, got $n"))
            new(Int(m), Int(n), colptr, rowval, nzval)
        end
    end
    SparseArrays.getcolptr(M::SSparseMatrixCSC) = M.colptr
    SparseArrays.rowvals(M::SSparseMatrixCSC) = M.rowval
end # @static

function SSparseMatrixCSC(
    m::Integer,
    n::Integer,
    colptr::SVector,
    rowval::SVector,
    nzval::SVector,
)
    Tv = eltype(nzval)
    Ti = promote_type(eltype(colptr), eltype(rowval))
    SSparseMatrixCSC{Tv,Ti,length(nzval),n + 1}(m, n, colptr, rowval, nzval)
end

function size(spm::SSparseMatrixCSC{Tv,Ti,NNZ,NP}, i::Integer) where {Tv,Ti,NNZ,NP}
    i == 1 ? spm.m : (i == 2 ? NP - 1 : throw(ArgumentError("dimension out of bound!")))
end
size(spm::SSparseMatrixCSC{Tv,Ti,NNZ,NP}) where {Tv,Ti,NNZ,NP} = (spm.m, NP - 1)

function getindex(ssp::SSparseMatrixCSC{Tv}, i, j) where {Tv}
    S = ssp.colptr[j]
    E = ssp.colptr[j+1] - 1
    for ii = S:E
        if i == ssp.rowval[ii]
            return ssp.nzval[ii]
        end
    end
    return Tv(0)
end

SparseArrays.issparse(::SSparseMatrixCSC) = true
SparseArrays.nonzeros(M::SSparseMatrixCSC) = M.nzval
SparseArrays.nnz(spm::SSparseMatrixCSC{Tv,Ti,NNZ}) where {Tv,Ti,NNZ} = NNZ
function SparseArrays.findnz(S::SSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    numnz = nnz(S)
    I = Vector{Ti}(undef, numnz)
    J = Vector{Ti}(undef, numnz)
    V = Vector{Tv}(undef, numnz)

    count = 1
    @inbounds for col = 1:S.n, k = S.colptr[col]:(S.colptr[col+1]-1)
        I[count] = S.rowval[k]
        J[count] = col
        V[count] = S.nzval[k]
        count += 1
    end

    return (I, J, V)
end
SparseArrays.dropzeros!(M::SSparseMatrixCSC; trim::Bool = false) = M
