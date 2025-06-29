# patches
LinearAlgebra.fzero(S::IMatrix) = zero(eltype(S))

Broadcast.BroadcastStyle(::Type{<:IMatrix}) = StructuredMatrixStyle{Diagonal}()

# specialize identity
Broadcast.broadcasted(
    ::AbstractArrayStyle{2},
    ::typeof(*),
    a::IMatrix{T},
    b::IMatrix,
) where {T} = IMatrix{T}(a.n)
Broadcast.broadcasted(
    ::AbstractArrayStyle{2},
    ::typeof(*),
    a::IMatrix,
    b::AbstractVecOrMat,
) = Diagonal(b)
Broadcast.broadcasted(
    ::AbstractArrayStyle{2},
    ::typeof(*),
    a::AbstractVecOrMat,
    b::IMatrix,
) = Diagonal(a)

Broadcast.broadcasted(
    ::AbstractArrayStyle{2},
    ::typeof(*),
    a::IMatrix,
    b::Number,
) = Diagonal(fill(b, a.n))
Broadcast.broadcasted(
    ::AbstractArrayStyle{2},
    ::typeof(*),
    a::Number,
    b::IMatrix,
) = Diagonal(fill(a, b.n))

# specialize perm matrix
function _broadcast_perm_prod(A::AbstractPermMatrix, B::AbstractMatrix)
    dest = similar(A, Base.promote_op(*, eltype(A), eltype(B)))
    @inbounds for (i, j, a) in IterNz(A)
        dest[i, j] = a * B[i, j]
    end
    return dest
end

Broadcast.broadcasted(
    ::AbstractArrayStyle{2},
    ::typeof(*),
    A::AbstractPermMatrix,
    B::AbstractMatrix,
) = _broadcast_perm_prod(A, B)
Broadcast.broadcasted(
    ::AbstractArrayStyle{2},
    ::typeof(*),
    A::AbstractMatrix,
    B::AbstractPermMatrix,
) = _broadcast_perm_prod(B, A)
Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), A::AbstractPermMatrix, B::AbstractPermMatrix) =
    _broadcast_perm_prod(A, B)

Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), A::AbstractPermMatrix, B::IMatrix) =
    Diagonal(A)
Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), A::IMatrix, B::AbstractPermMatrix) =
    Diagonal(B)

function _broadcast_diag_perm_prod(A::Diagonal, B::AbstractPermMatrix)
    Diagonal(A.diag .* getindex.(Ref(B), 1:size(A, 1), 1:size(A, 2)))
end

Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), A::AbstractPermMatrix, B::Diagonal) =
    _broadcast_diag_perm_prod(B, A)
Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), A::Diagonal, B::AbstractPermMatrix) =
    _broadcast_diag_perm_prod(A, B)

# TODO: commit this upstream
# specialize Diagonal .* SparseMatrixCSC
# Broadcast.broadcasted(
#     ::AbstractArrayStyle{2},
#     ::typeof(*),
#     A::Diagonal,
#     B::SparseMatrixCSC,
# ) = Broadcast.broadcasted(*, A, Diagonal(B))

# Broadcast.broadcasted(
#     ::AbstractArrayStyle{2},
#     ::typeof(*),
#     A::SparseMatrixCSC,
#     B::Diagonal,
# ) = Broadcast.broadcasted(*, Diagonal(A), B)

Broadcast.broadcasted(
    ::AbstractArrayStyle{2},
    ::typeof(*),
    a::AbstractPermMatrix,
    b::Number,
) = basetype(a)(a.perm, a.vals .* b)

Broadcast.broadcasted(
    ::AbstractArrayStyle{2},
    ::typeof(*),
    a::Number,
    b::AbstractPermMatrix,
) = basetype(b)(b.perm, a .* b.vals)
