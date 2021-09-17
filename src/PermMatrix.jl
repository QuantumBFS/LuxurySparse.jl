export PermMatrix, pmrand

"""
    PermMatrix{Tv, Ti}(perm::AbstractVector{Ti}, vals::AbstractVector{Tv}) where {Tv, Ti<:Integer}
    PermMatrix(perm::Vector{Ti}, vals::Vector{Tv}) where {Tv, Ti}
    PermMatrix(ds::AbstractMatrix)

PermMatrix represents a special kind linear operator: Permute and Multiply, which means `M * v = v[perm] * val`
Optimizations are used to make it much faster than `SparseMatrixCSC`.

* `perm` is the permutation order,
* `vals` is the multiplication factor.

[Generalized Permutation Matrix](https://en.wikipedia.org/wiki/Generalized_permutation_matrix)
"""
struct PermMatrix{Tv,Ti<:Integer,Vv<:AbstractVector{Tv},Vi<:AbstractVector{Ti}} <:
       AbstractMatrix{Tv}
    perm::Vi   # new orders
    vals::Vv   # multiplied values.

    function PermMatrix{Tv,Ti,Vv,Vi}(
        perm::Vi,
        vals::Vv,
    ) where {Tv,Ti<:Integer,Vv<:AbstractVector{Tv},Vi<:AbstractVector{Ti}}
        if length(perm) != length(vals)
            throw(
                DimensionMismatch(
                    "permutation ($(length(perm))) and multiply ($(length(vals))) length mismatch.",
                ),
            )
        end
        new{Tv,Ti,Vv,Vi}(perm, vals)
    end
end

function PermMatrix{Tv,Ti}(perm, vals) where {Tv,Ti<:Integer}
    PermMatrix{Tv,Ti,Vector{Tv},Vector{Ti}}(Vector{Ti}(perm), Vector{Tv}(vals))
end

function PermMatrix(
    perm::Vi,
    vals::Vv,
) where {Tv,Ti<:Integer,Vv<:AbstractVector{Tv},Vi<:AbstractVector{Ti}}
    PermMatrix{Tv,Ti,Vv,Vi}(perm, vals)
end

################# Array Functions ##################

size(M::PermMatrix) = (length(M.perm), length(M.perm))
function size(A::PermMatrix, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    elseif d <= 2
        return length(A.perm)
    else
        return 1
    end
end
getindex(M::PermMatrix{Tv}, i::Integer, j::Integer) where {Tv} =
    M.perm[i] == j ? M.vals[i] : zero(Tv)
function Base.setindex!(M::PermMatrix, val, i::Integer, j::Integer)
    if M.perm[i] == j
        @inbounds M.vals[i] = val
    else
        throw(BoundsError(M, (i, j)))
    end
end

copyto!(A::PermMatrix, B::PermMatrix) =
    (copyto!(A.perm, B.perm); copyto!(A.vals, B.vals); A)

"""
    pmrand(T::Type, n::Int) -> PermMatrix

Return random PermMatrix.
"""
function pmrand end

pmrand(::Type{T}, n::Int) where {T} = PermMatrix(randperm(n), randn(T, n))
pmrand(n::Int) = pmrand(Float64, n)

similar(x::PermMatrix{Tv,Ti}) where {Tv,Ti} =
    PermMatrix{Tv,Ti}(copy(x.perm), similar(x.vals))
similar(x::PermMatrix{Tv,Ti}, ::Type{T}) where {Tv,Ti,T} =
    PermMatrix{T,Ti}(copy(x.perm), similar(x.vals, T))

# TODO: rewrite this
# function show(io::IO, M::PermMatrix)
#     println("PermMatrix")
#     for item in zip(M.perm, M.vals)
#         i, p = item
#         println("- ($i) * $p")
#     end
# end

######### sparse array interfaces  #########
nnz(M::PermMatrix) = length(M.vals)
nonzeros(M::PermMatrix) = M.vals
findnz(M::PermMatrix) = (collect(1:size(M, 1)), M.perm, M.vals)
dropzeros!(M::PermMatrix; trim::Bool = false) = M
isdense(::PermMatrix) = false

Base.zero(pm::PermMatrix) = PermMatrix(pm.perm, zero(pm.vals))
