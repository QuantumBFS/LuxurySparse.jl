# IMatrix
for func in (:conj, :real, :transpose, :adjoint, :copy)
    @eval (Base.$func)(M::IMatrix{T}) where {T} = IMatrix{T}(M.n)
end
for func in (:adjoint!, :transpose!)
    @eval (LinearAlgebra.$func)(M::IMatrix) = M
end
Base.imag(M::IMatrix{T}) where {T} = Diagonal(zeros(T, M.n))

# PermMatrix
for func in (:conj, :real, :imag)
    @eval (Base.$func)(M::AbstractPermMatrix) = basetype(M)(M.perm, ($func)(M.vals))
end
Base.copy(M::AbstractPermMatrix) = basetype(M)(copy(M.perm), copy(M.vals))
Base.conj!(M::AbstractPermMatrix) = (conj!(M.vals); M)

function Base.transpose(M::AbstractPermMatrix)
    new_perm = fast_invperm(M.perm)
    return basetype(M)(new_perm, M.vals[new_perm])
end

Base.adjoint(S::AbstractPermMatrix{<:Real}) = transpose(S)
Base.adjoint(S::AbstractPermMatrix{<:Complex}) = conj!(transpose(S))

# scalar
Base.:*(A::IMatrix{T}, B::Number) where {T} = Diagonal(fill(promote_type(T, eltype(B))(B), A.n))
Base.:*(B::Number, A::IMatrix{T}) where {T} = Diagonal(fill(promote_type(T, eltype(B))(B), A.n))
Base.:/(A::IMatrix{T}, B::Number) where {T} =
    Diagonal(fill(promote_type(T, eltype(B))(1 / B), A.n))

Base.:*(A::AbstractPermMatrix, B::Number) = basetype(A)(A.perm, A.vals * B)
Base.:*(B::Number, A::AbstractPermMatrix) = A * B
Base.:/(A::AbstractPermMatrix, B::Number) = basetype(A)(A.perm, A.vals / B)
#+(A::PermMatrix, B::PermMatrix) = PermMatrix(A.dv+B.dv, A.ev+B.ev)
#-(A::PermMatrix, B::PermMatrix) = PermMatrix(A.dv-B.dv, A.ev-B.ev)

for op in [:+, :-]
    for MT in [:IMatrix, :AbstractPermMatrix]
        @eval begin
            # IMatrix, PermMatrix - SparseMatrixCSC
            Base.$op(A::$MT, B::SparseMatrixCSC) = $op(SparseMatrixCSC(A), B)
            Base.$op(B::SparseMatrixCSC, A::$MT) = $op(B, SparseMatrixCSC(A))
        end
    end
    @eval begin
        # IMatrix, PermMatrix - Diagonal
        Base.$op(d1::IMatrix, d2::Diagonal) = Diagonal($op(diag(d1), d2.diag))
        Base.$op(d1::Diagonal, d2::IMatrix) = Diagonal($op(d1.diag, diag(d2)))
        Base.$op(d1::AbstractPermMatrix, d2::Diagonal) = $op(SparseMatrixCSC(d1), d2)
        Base.$op(d1::Diagonal, d2::AbstractPermMatrix) = $op(d1, SparseMatrixCSC(d2))
        # PermMatrix - IMatrix
        Base.$op(A::AbstractPermMatrix, B::IMatrix) = $op(SparseMatrixCSC(A), SparseMatrixCSC(B))
        Base.$op(A::IMatrix, B::AbstractPermMatrix) = $op(SparseMatrixCSC(A), SparseMatrixCSC(B))
        Base.$op(A::AbstractPermMatrix, B::AbstractPermMatrix) = $op(SparseMatrixCSC(A), SparseMatrixCSC(B))
    end
end
# NOTE: promote to integer
Base.:+(d1::IMatrix{Ta}, d2::IMatrix{Tb}) where {Ta,Tb} =
    d1 == d2 ? Diagonal(fill(promote_type(Ta, Tb, Int)(2), d1.n)) : throw(DimensionMismatch())
Base.:-(d1::IMatrix{Ta}, d2::IMatrix{Tb}) where {Ta,Tb} =
    d1 == d2 ? spzeros(promote_type(Ta, Tb), d1.n, d1.n) : throw(DimensionMismatch())

for MT in [:IMatrix, :AbstractPermMatrix]
    @eval Base.:(==)(A::$MT, B::SparseMatrixCSC) = SparseMatrixCSC(A) == B
    @eval Base.:(==)(A::SparseMatrixCSC, B::$MT) = A == SparseMatrixCSC(B)
end
Base.:(==)(d1::IMatrix, d2::Diagonal) = all(isone, d2.diag)
Base.:(==)(d1::Diagonal, d2::IMatrix) = all(isone, d1.diag)
Base.:(==)(d1::AbstractPermMatrix, d2::Diagonal) = SparseMatrixCSC(d1) == SparseMatrixCSC(d2)
Base.:(==)(d1::Diagonal, d2::AbstractPermMatrix) = SparseMatrixCSC(d1) == SparseMatrixCSC(d2)
Base.:(==)(A::IMatrix, B::AbstractPermMatrix) = SparseMatrixCSC(A) == SparseMatrixCSC(B)
Base.:(==)(A::AbstractPermMatrix, B::IMatrix) = SparseMatrixCSC(A) == SparseMatrixCSC(B)

for MT in [:IMatrix, :AbstractPermMatrix]
    @eval Base.isapprox(A::$MT, B::SparseMatrixCSC; kwargs...) = isapprox(SparseMatrixCSC(A), B)
    @eval Base.isapprox(A::SparseMatrixCSC, B::$MT; kwargs...) = isapprox(A, SparseMatrixCSC(B))
    @eval Base.isapprox(d1::$MT, d2::Diagonal; kwargs...) = isapprox(diag(d1), d2.diag)
    @eval Base.isapprox(d1::Diagonal, d2::$MT; kwargs...) = isapprox(d1.diag, diag(d2))
end
Base.isapprox(A::IMatrix, B::AbstractPermMatrix; kwargs...) = isapprox(SparseMatrixCSC(A), SparseMatrixCSC(B); kwargs...)
Base.isapprox(A::AbstractPermMatrix, B::IMatrix; kwargs...) = isapprox(SparseMatrixCSC(A), SparseMatrixCSC(B); kwargs...)
