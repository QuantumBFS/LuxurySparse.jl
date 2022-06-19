struct IterNz{MT}
    A::MT
end

Base.length(nz::IterNz{<:AbstractMatrix}) = length(nz.A)
Base.length(nz::IterNz{<:AbstractSparseMatrix}) = nnz(nz.A)
Base.length(nz::IterNz{<:Adjoint}) = length(IterNz(nz.A.parent))
Base.length(nz::IterNz{<:Transpose}) = length(IterNz(nz.A.parent))
Base.length(nz::IterNz{<:Diagonal}) = size(nz.A, 1)
Base.length(nz::IterNz{<:PermMatrix}) = size(nz.A, 1)
Base.length(nz::IterNz{<:IMatrix}) = size(nz.A, 1)
Base.eltype(nz::IterNz) = eltype(nz.A)

# Diagonal
function Base.iterate(it::IterNz{<:Diagonal})
    0 == length(it) && return nothing
    return (1, 1, @inbounds it.A.diag[1]), 1
end
function Base.iterate(it::IterNz{<:Diagonal}, state)
    state == length(it) && return nothing
    state += 1
    return (state, state, @inbounds it.A.diag[state]), state
end

# IMatrix
function Base.iterate(it::IterNz{<:IMatrix{T}}) where T
    0 == length(it) && return nothing
    return (1, 1, one(T)), 1
end
function Base.iterate(it::IterNz{<:IMatrix{T}}, state) where T
    state == length(it) && return nothing
    state += 1
    return (state, state, one(T)), state
end

# PermMatrix
function Base.iterate(it::IterNz{<:PermMatrix})
    0 == length(it) && return nothing
    return (1, (@inbounds it.A.perm[1]), (@inbounds it.A.vals[1])), 1
end
function Base.iterate(it::IterNz{<:PermMatrix}, state)
    state == length(it) && return nothing
    state += 1
    return (state, (@inbounds it.A.perm[state]), (@inbounds it.A.vals[state])), state
end

# AbstractMatrix
function Base.iterate(it::IterNz{<:AbstractMatrix})
    0 == length(it) && return nothing
    return (1, 1, (@inbounds it.A[1])), (1, 1, 1)
end
function Base.iterate(it::IterNz{<:AbstractMatrix}, state)
    (i, j, k) = state
    k == length(it) && return nothing
    M = size(it.A, 1)
    if i == M
        i = 1
        j += 1
    else
        i += 1
    end
    k += 1
    return (i, j, (@inbounds it.A[k])), (i, j, k)
end

# SparseMatrixCSC
function Base.iterate(it::IterNz{<:SparseMatrixCSC})
    0 == length(it) && return nothing
    j = 1
    while j <= size(it.A, 2)
        it.A.colptr[j+1] > 1 && break
        j += 1
    end
    return (@inbounds(it.A.rowval[1]), j, @inbounds(it.A.nzval[1])), (j, 1)
end
function Base.iterate(it::IterNz{<:SparseMatrixCSC}, state)
    (j, k) = state
    k == length(it) && return nothing
    k += 1
    while j <= size(it.A, 2)
        it.A.colptr[j+1] > k && break
        j += 1
    end
    return (@inbounds(it.A.rowval[k]), j, @inbounds(it.A.nzval[k])), (j, k)
end

# Adjoint and Transpose
for (T, F) in [(:Adjoint, :conj), (:Transpose, :identity)]
    @eval function Base.iterate(it::IterNz{<:$T})
        res = iterate(IterNz(it.A.parent))
        res === nothing && return nothing
        (i, j, v), state = res
        (j, i, $F(v)), state
    end
    @eval function Base.iterate(it::IterNz{<:$T}, state)
        res = iterate(IterNz(it.A.parent), state)
        res === nothing && return nothing
        (i, j, v), state = res
        (j, i, $F(v)), state
    end
end