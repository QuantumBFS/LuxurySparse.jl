using FillArrays
using LuxurySparse
using SparseArrays
using SparseArrays.HigherOrderFns
using LinearAlgebra
using LinearAlgebra: StructuredMatrixStyle
using Base.Broadcast: BroadcastStyle, AbstractArrayStyle, Broadcasted, DefaultArrayStyle, materialize!

# patches
# TODO: commit this to upstream
LinearAlgebra.fzero(S::Matrix) = zero(eltype(S))
LinearAlgebra.fzero(S::PermMatrix) = zero(eltype(S))
LinearAlgebra.fzero(S::IMatrix) = zero(eltype(S))

# custom style
struct PermStyle <: AbstractArrayStyle{2} end

PermStyle(::Val{2}) = PermStyle()

Broadcast.BroadcastStyle(::Type{<:IMatrix}) = StructuredMatrixStyle{Diagonal}()
Broadcast.BroadcastStyle(::Type{<:PermMatrix}) = PermStyle()

Broadcast.BroadcastStyle(::PermStyle, ::HigherOrderFns.SPVM) = PermStyle()
Broadcast.BroadcastStyle(::PermStyle, ::LinearAlgebra.StructuredMatrixStyle{<:Diagonal}) = StructuredMatrixStyle{Diagonal}()

# specialize identity
Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), a::IMatrix{N, T}, b::IMatrix) where {N, T} = IMatrix{N, T}()
Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), a::IMatrix, b::AbstractVecOrMat) = Diagonal(b)
Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), a::AbstractVecOrMat, b::IMatrix) = Diagonal(a)

Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), a::IMatrix{S}, b::Number) where S = Diagonal(Fill(b, S))
Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), a::Number, b::IMatrix{S}) where S = Diagonal(Fill(a, S))

# TODO: commit this upstream
# specialize Diagonal .* SparseMatrixCSC
Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), A::Diagonal, B::SparseMatrixCSC) =
    Broadcast.broadcasted(*, A, Diagonal(B))

Broadcast.broadcasted(::AbstractArrayStyle{2}, ::typeof(*), A::SparseMatrixCSC, B::Diagonal) =
    Broadcast.broadcasted(*, Diagonal(A), B)

# Perm .* Perm
function Base.similar(bc::Broadcasted{PermStyle}, ::Type{ElType}) where ElType
    return _construct_perm_matrix(ElType, bc.args)
end

# create perm matrix based on the first perm matrix
_construct_perm_matrix(::Type{T}, args::Tuple) where T = _construct_perm_matrix(T, args...)
_construct_perm_matrix(::Type{T}, a::PermMatrix, xs...) where T = similar(a, T)
_construct_perm_matrix(::Type{T}, a, xs...) where T = _construct_perm_matrix(T, xs...)
_construct_perm_matrix(::Type, a) = nothing

function Base.copyto!(dest::PermMatrix, bc::Broadcasted{Nothing})
    if bc.f === identity && bc.args isa Tuple{AbstractArray}
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end

    i = 1
    for j in dest.perm
        dest[i, j] = bc[i, j]
        i += 1
    end
    return dest
end