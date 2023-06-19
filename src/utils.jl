function fast_invperm(order)
    v = similar(order)
    @inbounds @simd for i = 1:length(order)
        v[order[i]] = i
    end
    v
end

findnz(M::Diagonal) = (collect(1:size(M, 1)), collect(1:size(M, 1)), M.diag)
nnz(M::Diagonal) = length(M.diag)
