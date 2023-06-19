"""
    PermMatrix{Tv, Ti}(perm::AbstractVector{Ti}, vals::AbstractVector{Tv}) where {Tv, Ti<:Integer}
    PermMatrix(perm::Vector{Ti}, vals::Vector{Tv}) where {Tv, Ti}
    PermMatrix(ds::AbstractMatrix)

PermMatrix represents a special kind of linear operator: Permute and Multiply, which means `M * v = v[perm] * val`
Optimized implementations of `inv` and `*` make it much faster than `SparseMatrixCSC`.

* `perm` is the permutation order,
* `vals` is the multiplication factor.

[Generalized Permutation Matrix](https://en.wikipedia.org/wiki/Generalized_permutation_matrix)

# Example

```julia-repl
julia> PermMatrix([2,1,4,3], rand(4))
4×4 SDPermMatrix{Float64, Int64, Vector{Float64}, Vector{Int64}}:
 0.0       0.182251  0.0      0.0
 0.887485  0.0       0.0      0.0
 0.0       0.0       0.0      0.182831
 0.0       0.0       0.22895  0.0

```
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

Base.:(==)(d1::PermMatrix, d2::PermMatrix) = SparseMatrixCSC(d1) == SparseMatrixCSC(d2)
Base.isapprox(d1::PermMatrix, d2::PermMatrix; kwargs...) = isapprox(SparseMatrixCSC(d1), SparseMatrixCSC(d2); kwargs...)
Base.zero(pm::PermMatrix) = PermMatrix(pm.perm, zero(pm.vals))

################# Array Functions ##################

Base.size(M::PermMatrix) = (length(M.perm), length(M.perm))
function Base.size(A::PermMatrix, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    elseif d <= 2
        return length(A.perm)
    else
        return 1
    end
end
Base.getindex(M::PermMatrix{Tv}, i::Integer, j::Integer) where {Tv} =
    M.perm[i] == j ? M.vals[i] : zero(Tv)
function Base.setindex!(M::PermMatrix, val, i::Integer, j::Integer)
    if M.perm[i] == j
        @inbounds M.vals[i] = val
    else
        throw(BoundsError(M, (i, j)))
    end
end

Base.copyto!(A::PermMatrix, B::PermMatrix) =
    (copyto!(A.perm, B.perm); copyto!(A.vals, B.vals); A)

"""
    pmrand(T::Type, n::Int) -> PermMatrix

Return a random [`PermMatrix`](@ref) with type `T`, `n` rows and `n` columns.

# Example

```julia-repl
julia> pmrand(ComplexF64, 4)
4×4 SDPermMatrix{ComplexF64, Int64, Vector{ComplexF64}, Vector{Int64}}:
        0.0+0.0im      0.112104+0.0179632im       0.0+0.0im              0.0+0.0im
 -0.0625997+1.00664im       0.0+0.0im             0.0+0.0im              0.0+0.0im
        0.0+0.0im           0.0+0.0im             0.0+0.0im       -0.0981836-0.839471im
        0.0+0.0im           0.0+0.0im        0.735853-0.747084im         0.0+0.0im

```
"""
function pmrand end

pmrand(::Type{T}, n::Int) where {T} = PermMatrix(randperm(n), randn(T, n))
pmrand(n::Int) = pmrand(Float64, n)

Base.similar(x::PermMatrix{Tv,Ti}) where {Tv,Ti} =
    PermMatrix{Tv,Ti}(copy(x.perm), similar(x.vals))
Base.similar(x::PermMatrix{Tv,Ti}, ::Type{T}) where {Tv,Ti,T} =
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
findnz(M::PermMatrix) = (collect(1:size(M, 1)), M.perm, M.vals)
