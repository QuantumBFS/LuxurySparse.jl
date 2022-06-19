import Base: promote_rule

# SparseMatrixCSC
promote_rule(::Type{SparseMatrixCSC{Tv,Ti}}, ::Type{Matrix{T}}) where {Tv,Ti,T} =
    Matrix{promote_type(T, Tv)}

# IMatrix
promote_rule(
    ::Type{IMatrix{T}},
    ::Type{PermMatrix{Tv,Ti,Vv,Vi}},
) where {T,Tv,Ti,Vv,Vi} = (TT = promote_type(T, Tv); PermMatrix{TT,Ti,Vector{TT},Vi})
promote_rule(::Type{IMatrix{T}}, ::Type{SparseMatrixCSC{Tv,Ti}}) where {T,Tv,Ti} =
    SparseMatrixCSC{promote_type(T, Tv),Ti}
promote_rule(::Type{IMatrix{TA}}, ::Type{Matrix{TB}}) where {TA,TB} = Array{TB,2}

# PermMatrix
promote_rule(
    ::Type{PermMatrix{TvA,TiA}},
    ::Type{SparseMatrixCSC{TvB,TiB}},
) where {TvA,TiA,TvB,TiB} = SparseMatrixCSC{promote_type(TvA, TvB),promote_type(TiA, TiB)}
promote_rule(::Type{PermMatrix{Tv,Ti}}, ::Type{Matrix{T}}) where {Tv,Ti,T} =
    Array{promote_type(Tv, T),2}
