function irepeat(v::AbstractVector, n::Int)
    nV = length(v)
    res = similar(v, nV * n)
    @inbounds for j = 1:nV
        vj = v[j]
        base = (j - 1) * n
        @inbounds @simd for i = 1:n
            res[base+i] = vj
        end
    end
    res
end

function orepeat(v::AbstractVector, n::Int)
    nV = length(v)
    res = similar(v, nV * n)
    @inbounds for i = 1:n
        base = (i - 1) * nV
        @inbounds @simd for j = 1:nV
            res[base+j] = v[j]
        end
    end
    res
end


####### kronecker product ###########
fastkron(a, b) = kron(a, b)
fastkron(A::Diagonal{<:Number}, B::SparseMatrixCSC{<:Number}) = kron(PermMatrixCSC(A), B)
fastkron(A::SparseMatrixCSC{<:Number}, B::Diagonal{<:Number}) = kron(A, PermMatrixCSC(B))
fastkron(A::Diagonal{<:Number}, B::StridedMatrix{<:Number}) = kron(PermMatrixCSC(A), B)
fastkron(A::StridedMatrix{<:Number}, B::Diagonal{<:Number}) = kron(A, PermMatrixCSC(B))

LinearAlgebra.kron(A::IMatrix{Ta}, B::IMatrix{Tb}) where {Ta<:Number,Tb<:Number} =
    IMatrix{promote_type(Ta, Tb)}(A.n * B.n)
LinearAlgebra.kron(A::IMatrix{<:Number}, B::Diagonal{<:Number}) = A.n == 1 ? B : Diagonal(orepeat(B.diag, A.n))
LinearAlgebra.kron(B::Diagonal{<:Number}, A::IMatrix) = A.n == 1 ? B : Diagonal(irepeat(B.diag, A.n))

function LinearAlgebra.kron(A::AbstractMatrix{Tv}, B::IMatrix) where {Tv<:Number}
    B.n == 1 && return A
    mA, nA = size(A)
    nzval = Vector{Tv}(undef, B.n * mA * nA)
    rowval = Vector{Int}(undef, B.n * mA * nA)
    colptr = collect(1:mA:B.n*mA*nA+1)
    @inbounds for j = 1:nA
        source = view(A, :, j)
        startbase = (j - 1) * B.n * mA - mA
        for j2 = 1:B.n
            start = startbase + j2 * mA
            row = j2 - B.n
            @inbounds @simd for i = 1:mA
                nzval[start+i] = source[i]
                rowval[start+i] = row + B.n * i
            end
        end
    end
    SparseMatrixCSC(mA * B.n, nA * B.n, colptr, rowval, nzval)
end

function LinearAlgebra.kron(A::IMatrix, B::AbstractMatrix{Tv}) where {Tv<:Number}
    A.n == 1 && return B
    mB, nB = size(B)
    rowval = Vector{Int}(undef, nB * mB * A.n)
    nzval = Vector{Tv}(undef, nB * mB * A.n)
    @inbounds for j = 1:A.n
        r0 = (j - 1) * mB
        for j2 = 1:nB
            start = ((j - 1) * nB + j2 - 1) * mB
            @inbounds @simd for i = 1:mB
                rowval[start+i] = r0 + i
                nzval[start+i] = B[i, j2]
            end
        end
    end
    colptr = collect(1:mB:nB*mB*A.n+1)
    SparseMatrixCSC(mB * A.n, A.n * nB, colptr, rowval, nzval)
end

function LinearAlgebra.kron(A::IMatrix, B::SparseMatrixCSC{T}) where {T<:Number}
    A.n == 1 && return B
    mB, nB = size(B)
    nV = nnz(B)
    nzval = Vector{T}(undef, A.n * nV)
    rowval = Vector{Int}(undef, A.n * nV)
    colptr = Vector{Int}(undef, nB * A.n + 1)
    nzval = Vector{T}(undef, A.n * nV)
    colptr[1] = 1
    for i = 1:A.n
        r0 = (i - 1) * mB
        start = nV * (i - 1)
        @inbounds @simd for k = 1:nV
            rowval[start+k] = B.rowval[k] + r0
            nzval[start+k] = B.nzval[k]
        end
        colbase = (i - 1) * nB
        @inbounds @simd for j = 2:nB+1
            colptr[colbase+j] = B.colptr[j] + start
        end
    end
    SparseMatrixCSC(mB * A.n, nB * A.n, colptr, rowval, nzval)
end

function LinearAlgebra.kron(A::SparseMatrixCSC{T}, B::IMatrix) where {T<:Number}
    B.n == 1 && return A
    mA, nA = size(A)
    nV = nnz(A)
    rowval = Vector{Int}(undef, B.n * nV)
    colptr = Vector{Int}(undef, nA * B.n + 1)
    nzval = Vector{T}(undef, B.n * nV)
    z = 1
    colptr[1] = 1
    @inbounds for i = 1:nA
        rstart = A.colptr[i]
        rend = A.colptr[i+1] - 1
        colbase = (i - 1) * B.n + 1
        @inbounds for k = 1:B.n
            irow_Nb = k - B.n
            @inbounds @simd for r = rstart:rend
                rowval[z] = A.rowval[r] * B.n + irow_Nb
                nzval[z] = A.nzval[r]
                z += 1
            end
            colptr[colbase+k] = z
        end
    end
    SparseMatrixCSC(mA * B.n, nA * B.n, colptr, rowval, nzval)
end

function LinearAlgebra.kron(A::AbstractPermMatrix{T}, B::IMatrix) where {T<:Number}
    nA = size(A, 1)
    nB = size(B, 1)
    nB == 1 && return A
    vals = Vector{T}(undef, nB * nA)
    perm = Vector{Int}(undef, nB * nA)
    @inbounds for i = 1:nA
        start = (i - 1) * nB
        permAi = (A.perm[i] - 1) * nB
        val = A.vals[i]
        @inbounds @simd for j = 1:nB
            perm[start+j] = permAi + j
            vals[start+j] = val
        end
    end
    basetype(A)(perm, vals)
end

function LinearAlgebra.kron(A::IMatrix, B::AbstractPermMatrix{Tv,Ti}) where {Tv<:Number,Ti<:Integer}
    nA = size(A, 1)
    nB = size(B, 1)
    nA == 1 && return B
    perm = Vector{Int}(undef, nB * nA)
    vals = Vector{Tv}(undef, nB * nA)
    @inbounds for i = 1:nA
        start = (i - 1) * nB
        @inbounds @simd for j = 1:nB
            perm[start+j] = start + B.perm[j]
            vals[start+j] = B.vals[j]
        end
    end
    basetype(B)(perm, vals)
end

function LinearAlgebra.kron(A::StridedMatrix{Tv}, B::AbstractPermMatrix{Tb}) where {Tv<:Number,Tb<:Number}
    mA, nA = size(A)
    nB = size(B, 1)
    BC = PermMatrixCSC(B)
    perm, vals = BC.perm, BC.vals
    nzval = Vector{promote_type(Tv, Tb)}(undef, mA * nA * nB)
    rowval = Vector{Int}(undef, mA * nA * nB)
    colptr = collect(1:mA:nA*nB*mA+1)
    z = 1
    @inbounds for j = 1:nA
        @inbounds for j2 = 1:nB
            p2 = perm[j2]
            val2 = vals[j2]
            ir = p2
            @inbounds @simd for i = 1:mA
                nzval[z] = A[i, j] * val2  # merge
                rowval[z] = ir
                z += 1
                ir += nB
            end
        end
    end
    SparseMatrixCSC(mA * nB, nA * nB, colptr, rowval, nzval)
end

function LinearAlgebra.kron(A::AbstractPermMatrix{Ta}, B::StridedMatrix{Tb}) where {Tb<:Number,Ta<:Number}
    mB, nB = size(B)
    nA = size(A, 1)
    AC = PermMatrixCSC(A)
    perm, vals = AC.perm, AC.vals
    nzval = Vector{promote_type(Ta, Tb)}(undef, mB * nA * nB)
    rowval = Vector{Int}(undef, mB * nA * nB)
    colptr = collect(1:mB:nA*nB*mB+1)
    z = 0
    @inbounds for j = 1:nA
        p1 = perm[j]
        val2 = vals[j]
        ir = (p1 - 1) * mB
        for j2 = 1:nB
            @inbounds @simd for i2 = 1:mB
                nzval[z+i2] = B[i2, j2] * val2
                rowval[z+i2] = ir + i2
            end
            z += mB
        end
    end
    SparseMatrixCSC(nA * mB, nA * nB, colptr, rowval, nzval)
end

function LinearAlgebra.kron(A::AbstractPermMatrix{<:Number}, B::AbstractPermMatrix{<:Number})
    @assert basetype(A) == basetype(B)
    nA = size(A, 1)
    nB = size(B, 1)
    vals = kron(A.vals, B.vals)
    perm = Vector{Int}(undef, nB * nA)
    @inbounds for i = 1:nA
        start = (i - 1) * nB
        permAi = (A.perm[i] - 1) * nB
        @inbounds @simd for j = 1:nB
            perm[start+j] = permAi + B.perm[j]
        end
    end
    basetype(A)(perm, vals)
end

LinearAlgebra.kron(A::AbstractPermMatrix{<:Number}, B::Diagonal{<:Number}) = kron(A, basetype(A)(B))
LinearAlgebra.kron(A::Diagonal{<:Number}, B::AbstractPermMatrix{<:Number}) = kron(basetype(B)(A), B)

function LinearAlgebra.kron(A::AbstractPermMatrix{Ta}, B::SparseMatrixCSC{Tb}) where {Ta<:Number,Tb<:Number}
    nA = size(A, 1)
    mB, nB = size(B)
    nV = nnz(B)
    AC = PermMatrixCSC(A)
    perm, vals = AC.perm, AC.vals
    nzval = Vector{promote_type(Ta, Tb)}(undef, nA * nV)
    rowval = Vector{Int}(undef, nA * nV)
    colptr = Vector{Int}(undef, nA * nB + 1)
    colptr[1] = 1
    @inbounds @simd for i = 1:nA
        start_row = (i - 1) * nV
        start_ri = (perm[i] - 1) * mB
        v0 = vals[i]
        @inbounds @simd for j = 1:nV
            nzval[start_row+j] = B.nzval[j] * v0
            rowval[start_row+j] = B.rowval[j] + start_ri
        end
        start_col = (i - 1) * nB + 1
        start_ci = (i - 1) * nV
        @inbounds @simd for j = 1:nB
            colptr[start_col+j] = B.colptr[j+1] + start_ci
        end
    end
    SparseMatrixCSC(mB * nA, nB * nA, colptr, rowval, nzval)
end

function LinearAlgebra.kron(A::SparseMatrixCSC{T}, B::AbstractPermMatrix{Tb}) where {T<:Number,Tb<:Number}
    nB = size(B, 1)
    mA, nA = size(A)
    nV = nnz(A)
    BC = PermMatrixCSC(B)
    perm, vals = BC.perm, BC.vals
    rowval = Vector{Int}(undef, nB * nV)
    colptr = Vector{Int}(undef, nA * nB + 1)
    nzval = Vector{promote_type(T, Tb)}(undef, nB * nV)
    z = 1
    colptr[z] = 1
    @inbounds for i = 1:nA
        rstart = A.colptr[i]
        rend = A.colptr[i+1] - 1
        @inbounds for k = 1:nB
            irow = perm[k]
            bval = vals[k]
            irow_nB = irow - nB
            @inbounds @simd for r = rstart:rend
                rowval[z] = A.rowval[r] * nB + irow_nB
                nzval[z] = A.nzval[r] * bval
                z += 1
            end
            colptr[(i-1)*nB+k+1] = z
        end
    end
    SparseMatrixCSC(mA * nB, nA * nB, colptr, rowval, nzval)
end
