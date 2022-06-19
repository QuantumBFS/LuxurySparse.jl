import Base: conj, copy, real, imag
import LinearAlgebra: transpose, transpose!, adjoint!, adjoint

# IMatrix
for func in (:conj, :real, :transpose, :adjoint, :copy)
    @eval ($func)(M::IMatrix{T}) where {T} = IMatrix{T}(M.n)
end
for func in (:adjoint!, :transpose!)
    @eval ($func)(M::IMatrix) = M
end
imag(M::IMatrix{T}) where {N,T} = Diagonal(zeros(T, M.n))

# PermMatrix
for func in (:conj, :real, :imag)
    @eval ($func)(M::PermMatrix) = PermMatrix(M.perm, ($func)(M.vals))
end
copy(M::PermMatrix) = PermMatrix(copy(M.perm), copy(M.vals))

function transpose(M::PermMatrix)
    new_perm = fast_invperm(M.perm)
    return PermMatrix(new_perm, M.vals[new_perm])
end

adjoint(S::PermMatrix{<:Real}) = transpose(S)
adjoint(S::PermMatrix{<:Complex}) = conj(transpose(S))

# scalar
import Base: *, /, ==, +, -, ≈
*(A::IMatrix{T}, B::Number) where {T} = Diagonal(fill(promote_type(T, eltype(B))(B), A.n))
*(B::Number, A::IMatrix{T}) where {T} = Diagonal(fill(promote_type(T, eltype(B))(B), A.n))
/(A::IMatrix{T}, B::Number) where {T} =
    Diagonal(fill(promote_type(T, eltype(B))(1 / B), A.n))

*(A::PermMatrix, B::Number) = PermMatrix(A.perm, A.vals * B)
*(B::Number, A::PermMatrix) = A * B
/(A::PermMatrix, B::Number) = PermMatrix(A.perm, A.vals / B)
#+(A::PermMatrix, B::PermMatrix) = PermMatrix(A.dv+B.dv, A.ev+B.ev)
#-(A::PermMatrix, B::PermMatrix) = PermMatrix(A.dv-B.dv, A.ev-B.ev)

for op in [:+, :-]
    for MT in [:IMatrix, :PermMatrix]
        @eval begin
            # IMatrix, PermMatrix - SparseMatrixCSC
            $op(A::$MT, B::SparseMatrixCSC) = $op(SparseMatrixCSC(A), B)
            $op(B::SparseMatrixCSC, A::$MT) = $op(B, SparseMatrixCSC(A))
        end
    end
    @eval begin
        # IMatrix, PermMatrix - Diagonal
        $op(d1::IMatrix, d2::Diagonal) = Diagonal($op(diag(d1), d2.diag))
        $op(d1::Diagonal, d2::IMatrix) = Diagonal($op(d1.diag, diag(d2)))
        $op(d1::PermMatrix, d2::Diagonal) = $op(SparseMatrixCSC(d1), d2)
        $op(d1::Diagonal, d2::PermMatrix) = $op(d1, SparseMatrixCSC(d2))
        # PermMatrix - IMatrix
        $op(A::PermMatrix, B::IMatrix) = $op(SparseMatrixCSC(A), SparseMatrixCSC(B))
        $op(A::IMatrix, B::PermMatrix) = $op(SparseMatrixCSC(A), SparseMatrixCSC(B))
        $op(A::PermMatrix, B::PermMatrix) = $op(SparseMatrixCSC(A), SparseMatrixCSC(B))
    end
end
# NOTE: promote to integer
+(d1::IMatrix{Ta}, d2::IMatrix{Tb}) where {Ta,Tb} =
    d1 == d2 ? Diagonal(fill(promote_type(Ta, Tb, Int)(2), d1.n)) : throw(DimensionMismatch())
-(d1::IMatrix{Ta}, d2::IMatrix{Tb}) where {Ta,Tb} =
    d1 == d2 ? spzeros(promote_type(Ta, Tb), d1.n, d1.n) : throw(DimensionMismatch())

for MT in [:IMatrix, :PermMatrix]
    @eval Base.:(==)(A::$MT, B::SparseMatrixCSC) = SparseMatrixCSC(A) == B
    @eval Base.:(==)(A::SparseMatrixCSC, B::$MT) = A == SparseMatrixCSC(B)
end
Base.:(==)(d1::IMatrix, d2::Diagonal) = all(isone, d2.diag)
Base.:(==)(d1::Diagonal, d2::IMatrix) = all(isone, d1.diag)
Base.:(==)(d1::PermMatrix, d2::Diagonal) = SparseMatrixCSC(d1) == SparseMatrixCSC(d2)
Base.:(==)(d1::Diagonal, d2::PermMatrix) = SparseMatrixCSC(d1) == SparseMatrixCSC(d2)
Base.:(==)(A::IMatrix, B::PermMatrix) = SparseMatrixCSC(A) == SparseMatrixCSC(B)
Base.:(==)(A::PermMatrix, B::IMatrix) = SparseMatrixCSC(A) == SparseMatrixCSC(B)

for MT in [:IMatrix, :PermMatrix]
    @eval Base.isapprox(A::$MT, B::SparseMatrixCSC; kwargs...) = isapprox(SparseMatrixCSC(A), B)
    @eval Base.isapprox(A::SparseMatrixCSC, B::$MT; kwargs...) = isapprox(A, SparseMatrixCSC(B))
    @eval Base.isapprox(d1::$MT, d2::Diagonal; kwargs...) = isapprox(diag(d1), d2.diag)
    @eval Base.isapprox(d1::Diagonal, d2::$MT; kwargs...) = isapprox(d1.diag, diag(d2))
end
Base.isapprox(A::IMatrix, B::PermMatrix; kwargs...) = isapprox(SparseMatrixCSC(A), SparseMatrixCSC(B); kwargs...)
Base.isapprox(A::PermMatrix, B::IMatrix; kwargs...) = isapprox(SparseMatrixCSC(A), SparseMatrixCSC(B); kwargs...)