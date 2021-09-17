import Base: inv
import LinearAlgebra: det, diag, logdet, mul!, lmul!, rmul!, rdiv!
using LinearAlgebra

####### linear algebra  ######
inv(M::IMatrix) = M
det(M::IMatrix) = 1
diag(M::IMatrix{N,T}) where {N,T} = ones(T, N)
logdet(M::IMatrix) = 0
Base.sqrt(x::PermMatrix) = sqrt(Matrix(x))
Base.sqrt(x::IMatrix) = x
Base.exp(x::PermMatrix) = exp(Matrix(x))
Base.exp(x::IMatrix) = ℯ * x

#det(M::PermMatrix) = parity(M.perm)*prod(M.vals)
function inv(M::PermMatrix)
    new_perm = fast_invperm(M.perm)
    return PermMatrix(new_perm, 1.0 ./ M.vals[new_perm])
end

####### multiply ###########
*(A::IMatrix{N}, B::AbstractVector) where {N} =
    size(A, 2) == size(B, 1) ? B :
    throw(
        DimensionMismatch(
            "matrix A has dimensions $(size(A)), matrix B has dimensions $((size(B, 1), 1))",
        ),
    )

for MATTYPE in
    [:AbstractMatrix, :StridedMatrix, :Diagonal, :SparseMatrixCSC, :Matrix, :PermMatrix]
    @eval *(A::IMatrix{N}, B::$MATTYPE) where {N} =
        N == size(B, 1) ? B :
        throw(
            DimensionMismatch(
                "matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))",
            ),
        )

    @eval *(A::$MATTYPE, B::IMatrix{N}) where {N} =
        size(A, 2) == N ? A :
        throw(
            DimensionMismatch(
                "matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))",
            ),
        )
end

*(A::Adjoint{T,<:AbstractVector{T}}, B::IMatrix) where {T} =
    size(A, 2) == size(B, 1) ? A :
    throw(
        DimensionMismatch(
            "matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))",
        ),
    )

*(A::IMatrix, B::IMatrix) =
    size(A, 2) == size(B, 1) ? A :
    throw(
        DimensionMismatch(
            "matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))",
        ),
    )


########## Multiplication #############

# NOTE: making them dry?
# to vector
function *(A::PermMatrix{Ta}, X::AbstractVector{Tx}) where {Ta,Tx}
    nX = length(X)
    nX == size(A, 2) || throw(DimensionMismatch())
    v = similar(X, promote_type(Ta, Tx))
    @simd for i = 1:nX
        @inbounds v[i] = A.vals[i] * X[A.perm[i]]
    end
    v
end

function *(X::Adjoint{Tx,<:AbstractVector{Tx}}, A::PermMatrix{Ta}) where {Tx,Ta}
    nX = length(X)
    nX == size(A, 1) || throw(DimensionMismatch())
    v = similar(X, promote_type(Tx, Ta))
    @simd for i = 1:nX
        @inbounds v[A.perm[i]] = A.vals[i] * X[i]
    end
    v
end

# to diagonal
function *(D::Diagonal{Td}, A::PermMatrix{Ta}) where {Td,Ta}
    T = Base.promote_op(*, Td, Ta)
    PermMatrix(A.perm, A.vals .* D.diag)
end

function *(A::PermMatrix{Ta}, D::Diagonal{Td}) where {Td,Ta}
    T = Base.promote_op(*, Td, Ta)
    PermMatrix(A.perm, A.vals .* view(D.diag, A.perm))
end

# to self
function *(A::PermMatrix, B::PermMatrix)
    size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    PermMatrix(B.perm[A.perm], A.vals .* view(B.vals, A.perm))
end

# to matrix
function *(A::PermMatrix, X::AbstractMatrix)
    size(X, 1) == size(A, 2) || throw(DimensionMismatch())
    return @views A.vals .* X[A.perm, :]   # this may be inefficient for sparse CSC matrix.
end

function *(X::AbstractMatrix, A::PermMatrix)
    mX, nX = size(X)
    nX == size(A, 1) || throw(DimensionMismatch())
    return @views (A.vals'.*X)[:, fast_invperm(A.perm)]
end

# NOTE: this is just a temperory fix for v0.7. We should overload mul! in
# the future (when we start to drop v0.6) to enable buildin lazy evaluation.

*(x::Adjoint{<:Any,<:AbstractVector}, D::PermMatrix) = Matrix(x) * D
*(x::Transpose{<:Any,<:AbstractVector}, D::PermMatrix) = Matrix(x) * D
*(A::Adjoint{<:Any,<:AbstractArray}, D::PermMatrix) = Adjoint(adjoint(D) * parent(A))
*(A::Transpose{<:Any,<:AbstractArray}, D::PermMatrix) = Transpose(transpose(D) * parent(A))
*(A::Adjoint{<:Any,<:PermMatrix}, D::PermMatrix) = adjoint(parent(A)) * D
*(A::Transpose{<:Any,<:PermMatrix}, D::PermMatrix) = transpose(parent(A)) * D
*(A::PermMatrix, D::Adjoint{<:Any,<:PermMatrix}) = A * adjoint(parent(D))
*(A::PermMatrix, D::Transpose{<:Any,<:PermMatrix}) = A * transpose(parent(D))

# for MAT in [:AbstractArray, :Matrix, :SparseMatrixCSC, :PermMatrix]
#     @eval begin
#         *(A::Adjoint{<:Any, <:$MAT}, D::PermMatrix) = copy(A) * D
#         *(A::Transpose{<:Any, <:$MAT}, D::PermMatrix) = copy(A) * D
#         *(A::PermMatrix, D::Adjoint{<:Any, <:$MAT}) = A * copy(D)
#         *(A::PermMatrix, D::Transpose{<:Any, <:$MAT}) = A * copy(D)
#     end
# end

############### Transpose, Adjoint for IMatrix ###############
for MAT in
    [:AbstractArray, :AbstractVector, :Matrix, :SparseMatrixCSC, :PermMatrix, :IMatrix]
    @eval *(A::Adjoint{<:Any,<:$MAT}, D::IMatrix) = Adjoint(D * parent(A))
    @eval *(A::Transpose{<:Any,<:$MAT}, D::IMatrix) = Transpose(D * parent(A))
    if MAT != :AbstactVector
        @eval *(A::IMatrix, D::Transpose{<:Any,<:$MAT}) = Transpose(parent(D) * A)
        @eval *(A::IMatrix, D::Adjoint{<:Any,<:$MAT}) = Adjoint(parent(D) * A)
    end
end

# to sparse
function *(A::PermMatrix, X::SparseMatrixCSC)
    nA = size(A, 1)
    mX, nX = size(X)
    mX == nA || throw(DimensionMismatch())
    perm = fast_invperm(A.perm)
    nzval = similar(X.nzval)
    rowval = similar(X.rowval)
    @inbounds for j = 1:nX
        @inbounds for k = X.colptr[j]:X.colptr[j+1]-1
            r = perm[X.rowval[k]]
            nzval[k] = X.nzval[k] * A.vals[r]
            rowval[k] = r
        end
    end
    sp = SparseMatrixCSC(mX, nX, X.colptr, rowval, nzval)
    SparseMatrixCSC(sp')'
end

function *(X::SparseMatrixCSC, A::PermMatrix)
    nA = size(A, 1)
    mX, nX = size(X)
    nX == nA || throw(DimensionMismatch())
    perm = fast_invperm(A.perm)
    nzval = similar(X.nzval)
    colptr = similar(X.colptr)
    rowval = similar(X.rowval)
    colptr[1] = 1
    z = 1
    @inbounds for j = 1:nA
        pk = perm[j]
        va = A.vals[pk]
        @inbounds @simd for k = X.colptr[pk]:X.colptr[pk+1]-1
            nzval[z] = X.nzval[k] * va
            rowval[z] = X.rowval[k]
            z += 1
        end
        colptr[j+1] = z
    end
    SparseMatrixCSC(mX, nX, colptr, rowval, nzval)
end

############ SparseMatrixCSC and Diagonal #############
function mul!(Y::SparseMatrixCSC, X::SparseMatrixCSC, A::Diagonal{T,Vector{T}}) where {T}
    nA = size(A, 1)
    mX, nX = size(X)
    nX == nA || throw(DimensionMismatch())
    @inbounds for j = 1:nA
        for k = X.colptr[j]:X.colptr[j+1]-1
            Y.nzval[k] *= A.diag[j]
        end
    end
    Y
end

function mul!(Y::SparseMatrixCSC, A::Diagonal{T,Vector{T}}, X::SparseMatrixCSC) where {T}
    nA = size(A, 2)
    mX, nX = size(X)
    mX == nA || throw(DimensionMismatch())
    @inbounds @simd for i = 1:nnz(X)
        Y.nzval[i] *= A.diag[X.rowval[i]]
    end
    Y
end


rmul!(A::SparseMatrixCOO, B::Int) = (A.vs *= B; A)
lmul!(B::Int, A::SparseMatrixCOO) = (A.vs *= B; A)
rdiv!(A::SparseMatrixCOO, B::Int) = (A.vs /= B; A)

*(A::SparseMatrixCOO, B::Int) = rmul!(copy(A), B)
*(B::Int, A::SparseMatrixCOO) = lmul!(B, copy(A))
/(A::SparseMatrixCOO, B::Int) = rdiv!(copy(A), B)

-(ii::IMatrix) = (-1) * ii
-(pm::PermMatrix) = (-1) * pm

for FUNC in [:randn!, :rand!]
    @eval function Random.$FUNC(m::Diagonal)
        $FUNC(m.diag)
        return m
    end

    @eval function Random.$FUNC(m::SparseMatrixCSC)
        $FUNC(m.nzval)
        return m
    end

    @eval function Random.$FUNC(m::PermMatrix)
        $FUNC(m.vals)
        return m
    end
end
