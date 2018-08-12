"""
    notdense(M) -> Bool

Return true if a matrix is not dense.

Note:
It is not exactly same as isparse, e.g. Diagonal, IMatrix and PermMatrix are both notdense but not isparse.
"""
function notdense end

notdense(M)::Bool = issparse(M)
@static if VERSION >= v"0.7-"
notdense(x::Transpose) = notdense(parent(x))
notdense(x::Adjoint) = notdense(parent(x))
end

"""faster invperm"""
function fast_invperm(order)
    v = similar(order)
    @inbounds @simd for i=1:length(order)
        v[order[i]] = i
    end
    v
end

dropzeros!(A::Diagonal) = A
