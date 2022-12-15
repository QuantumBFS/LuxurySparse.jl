####### linear algebra  ######
Base.inv(M::IMatrix) = M
LinearAlgebra.det(M::IMatrix) = 1
LinearAlgebra.diag(M::IMatrix{T}) where {T} = ones(T, M.n)
LinearAlgebra.logdet(M::IMatrix) = 0
Base.sqrt(x::PermMatrix) = sqrt(Matrix(x))
Base.sqrt(x::IMatrix) = x
Base.exp(x::PermMatrix) = exp(Matrix(x))
Base.exp(x::IMatrix) = ℯ * x

#det(M::PermMatrix) = parity(M.perm)*prod(M.vals)
function Base.inv(M::PermMatrix)
    new_perm = fast_invperm(M.perm)
    return PermMatrix(new_perm, 1.0 ./ M.vals[new_perm])
end

####### multiply ###########
Base.:*(A::IMatrix, B::AbstractVector) =
    size(A, 2) == size(B, 1) ? B :
    throw(
        DimensionMismatch(
            "matrix A has dimensions $(size(A)), matrix B has dimensions $((size(B, 1), 1))",
        ),
    )

for MATTYPE in
    [:AbstractMatrix, :StridedMatrix, :Diagonal, :SparseMatrixCSC, :Matrix, :PermMatrix]
    @eval Base.:*(A::IMatrix, B::$MATTYPE) =
        A.n == size(B, 1) ? B :
        throw(
            DimensionMismatch(
                "matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))",
            ),
        )

    @eval Base.:*(A::$MATTYPE, B::IMatrix) =
        size(A, 2) == B.n ? A :
        throw(
            DimensionMismatch(
                "matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))",
            ),
        )
end

Base.:*(A::Adjoint{T,<:AbstractVector{T}}, B::IMatrix) where {T} =
    size(A, 2) == size(B, 1) ? A :
    throw(
        DimensionMismatch(
            "matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))",
        ),
    )

Base.:*(A::IMatrix, B::IMatrix) =
    size(A, 2) == size(B, 1) ? A :
    throw(
        DimensionMismatch(
            "matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))",
        ),
    )


########## Multiplication #############

function LinearAlgebra.mul!(Y::AbstractVector, A::PermMatrix, X::AbstractVector, alpha::Number, beta::Number)
    length(X) == size(A, 2) || throw(DimensionMismatch("input X length does not match PermMatrix A"))
    length(Y) == size(A, 2) || throw(DimensionMismatch("output Y length does not match PermMatrix A"))

    @inbounds for I in eachindex(X)
        Y[I] = A.vals[I] * X[A.perm[I]] * alpha + beta * Y[I]
    end
    return Y
end

# to diagonal
function Base.:*(D::Diagonal{Td}, A::PermMatrix{Ta}) where {Td,Ta}
    PermMatrix(A.perm, A.vals .* D.diag)
end

function Base.:*(A::PermMatrix{Ta}, D::Diagonal{Td}) where {Td,Ta}
    PermMatrix(A.perm, A.vals .* view(D.diag, A.perm))
end

# to self
function Base.:*(A::PermMatrix, B::PermMatrix)
    size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    PermMatrix(B.perm[A.perm], A.vals .* view(B.vals, A.perm))
end

# to matrix
function Base.:*(A::PermMatrix, X::AbstractMatrix)
    size(X, 1) == size(A, 2) || throw(DimensionMismatch())
    return A.vals .* view(X,A.perm,:)   # this may be inefficient for sparse CSC matrix.
end

function Base.:*(X::AbstractMatrix, A::PermMatrix)
    mX, nX = size(X)
    nX == size(A, 1) || throw(DimensionMismatch())
    perm = fast_invperm(A.perm)
    return transpose(view(A.vals, perm)) .* view(X, :, perm)
end

# NOTE: this is just a temperory fix for v0.7. We should overload mul! in
# the future (when we start to drop v0.6) to enable buildin lazy evaluation.

Base.:*(x::Adjoint{<:Any,<:AbstractVector}, D::PermMatrix) = Matrix(x) * D
Base.:*(x::Transpose{<:Any,<:AbstractVector}, D::PermMatrix) = Matrix(x) * D
Base.:*(A::Adjoint{<:Any,<:AbstractArray}, D::PermMatrix) = Adjoint(adjoint(D) * parent(A))
Base.:*(A::Transpose{<:Any,<:AbstractArray}, D::PermMatrix) = Transpose(transpose(D) * parent(A))
Base.:*(A::Adjoint{<:Any,<:PermMatrix}, D::PermMatrix) = adjoint(parent(A)) * D
Base.:*(A::Transpose{<:Any,<:PermMatrix}, D::PermMatrix) = transpose(parent(A)) * D
Base.:*(A::PermMatrix, D::Adjoint{<:Any,<:PermMatrix}) = A * adjoint(parent(D))
Base.:*(A::PermMatrix, D::Transpose{<:Any,<:PermMatrix}) = A * transpose(parent(D))

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
    @eval Base.:*(A::Adjoint{<:Any,<:$MAT}, D::IMatrix) = Adjoint(D * parent(A))
    @eval Base.:*(A::Transpose{<:Any,<:$MAT}, D::IMatrix) = Transpose(D * parent(A))
    if MAT != :AbstactVector
        @eval Base.:*(A::IMatrix, D::Transpose{<:Any,<:$MAT}) = Transpose(parent(D) * A)
        @eval Base.:*(A::IMatrix, D::Adjoint{<:Any,<:$MAT}) = Adjoint(parent(D) * A)
    end
end

# to sparse
function Base.:*(A::PermMatrix, X::SparseMatrixCSC)
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

function Base.:*(X::SparseMatrixCSC, A::PermMatrix)
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

LinearAlgebra.rmul!(A::SparseMatrixCOO, B::Int) = (A.vs *= B; A)
LinearAlgebra.lmul!(B::Int, A::SparseMatrixCOO) = (A.vs *= B; A)
LinearAlgebra.rdiv!(A::SparseMatrixCOO, B::Int) = (A.vs /= B; A)

Base.:*(A::SparseMatrixCOO, B::Int) = rmul!(copy(A), B)
Base.:*(B::Int, A::SparseMatrixCOO) = lmul!(B, copy(A))
Base.:/(A::SparseMatrixCOO, B::Int) = rdiv!(copy(A), B)

Base.:-(ii::IMatrix) = (-1) * ii
Base.:-(pm::PermMatrix) = (-1) * pm

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
