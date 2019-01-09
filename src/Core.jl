"""
    isdense(M) -> Bool

Return true if a matrix is dense.

Note:
It is not exactly same as !isparse, e.g. Diagonal, IMatrix and PermMatrix are both not isdense and not isparse.
"""
function isdense end

isdense(M)::Bool = !issparse(M)
isdense(::Diagonal) = false
isdense(x::Transpose) = isdense(parent(x))
isdense(x::Adjoint) = isdense(parent(x))

"""faster invperm"""
function fast_invperm(order)
    v = similar(order)
    @inbounds @simd for i=1:length(order)
        v[order[i]] = i
    end
    v
end

dropzeros!(A::Diagonal; trim::Bool=false) = A
