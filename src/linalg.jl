# inversion
inv(A::IMatrix) = A
det(A::IMatrix) = 1
diag(A::IMatrix{N, T}) where {N, T} = ones(T, N)
logdet(A::IMatrix) = 0

function inv(M::PermMatrix)
    new_perm = fast_invperm(M.perm)
    return PermMatrix(new_perm, 1.0./M.vals[new_perm])
end
