abstract type AbstractPermMatrix{Tv, Ti} <: AbstractMatrix{Tv} end
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
       AbstractPermMatrix{Tv,Ti}
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
basetype(pm::PermMatrix) = PermMatrix
Base.getindex(M::PermMatrix{Tv}, i::Integer, j::Integer) where {Tv} =
    M.perm[i] == j ? M.vals[i] : zero(Tv)
function Base.setindex!(M::PermMatrix, val, i::Integer, j::Integer)
    @assert M.perm[i] == j "Can not set index due to the absense of entry: ($i, $j)"
    @inbounds M.vals[i] = val
end

# the column major version of `PermMatrix`
struct PermMatrixCSC{Tv,Ti<:Integer,Vv<:AbstractVector{Tv},Vi<:AbstractVector{Ti}} <:
       AbstractPermMatrix{Tv,Ti}
    perm::Vi   # new orders
    vals::Vv   # multiplied values.

    function PermMatrixCSC{Tv,Ti,Vv,Vi}(
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
basetype(pm::PermMatrixCSC) = PermMatrixCSC
@propagate_inbounds function Base.getindex(M::PermMatrixCSC{Tv}, i::Integer, j::Integer) where {Tv}
    @boundscheck 0 < j <= size(M, 2)
    @inbounds M.perm[j] == i ? M.vals[j] : zero(Tv)
end
function Base.setindex!(M::PermMatrixCSC, val, i::Integer, j::Integer)
    @assert M.perm[j] == i "Can not set index due to the absense of entry: ($i, $j)"
    @inbounds M.vals[j] = val
end

for MT in [:PermMatrix, :PermMatrixCSC]
    @eval begin
        function $MT{Tv,Ti}(perm, vals) where {Tv,Ti<:Integer}
            $MT{Tv,Ti,Vector{Tv},Vector{Ti}}(Vector{Ti}(perm), Vector{Tv}(vals))
        end

        function $MT(
            perm::Vi,
            vals::Vv,
        ) where {Tv,Ti<:Integer,Vv<:AbstractVector{Tv},Vi<:AbstractVector{Ti}}
            $MT{Tv,Ti,Vv,Vi}(perm, vals)
        end
    end
end
Base.zero(pm::AbstractPermMatrix) = basetype(pm)(pm.perm, zero(pm.vals))
Base.similar(x::AbstractPermMatrix{Tv,Ti}) where {Tv,Ti} =
    typeof(x)(copy(x.perm), similar(x.vals))
Base.similar(x::AbstractPermMatrix{Tv,Ti}, ::Type{T}) where {Tv,Ti,T} =
    basetype(x){T,Ti}(copy(x.perm), similar(x.vals, T))

################# Comparison ##################
Base.:(==)(d1::AbstractPermMatrix, d2::AbstractPermMatrix) = SparseMatrixCSC(d1) == SparseMatrixCSC(d2)
Base.isapprox(d1::AbstractPermMatrix, d2::AbstractPermMatrix; kwargs...) = isapprox(SparseMatrixCSC(d1), SparseMatrixCSC(d2); kwargs...)
Base.copyto!(A::AbstractPermMatrix, B::AbstractPermMatrix) =
    (copyto!(A.perm, B.perm); copyto!(A.vals, B.vals); A)

################# Array Functions ##################

Base.size(M::AbstractPermMatrix) = (length(M.perm), length(M.perm))
function Base.size(A::AbstractPermMatrix, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    elseif d <= 2
        return length(A.perm)
    else
        return 1
    end
end

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

pmcscrand(::Type{T}, n::Int) where {T} = PermMatrixCSC(randperm(n), randn(T, n))
pmcscrand(n::Int) = pmcscrand(Float64, n)

Base.show(io::IO, ::MIME"text/plain", M::AbstractPermMatrix) = show(io, M)
function Base.show(io::IO, M::AbstractPermMatrix)
    n = size(M, 1)
    println(io, typeof(M))
    nmax = 20
    for (k, (i, j, p)) in enumerate(IterNz(M))
        if k <= nmax || k > n-nmax
            print(io, "($i, $j) = $p")
            k < n && println(io)
        elseif k == nmax+1
            println(io, "...")
        end
    end
end
Base.hash(pm::AbstractPermMatrix) = hash((pm.perm, pm.vals))

######### sparse array interfaces  #########
nnz(M::AbstractPermMatrix) = length(M.vals)
findnz(M::PermMatrix) = (collect(1:size(M, 1)), M.perm, M.vals)
findnz(M::PermMatrixCSC) = (M.perm, collect(1:size(M, 1)), M.vals)
