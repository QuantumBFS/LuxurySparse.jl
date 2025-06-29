function fast_invperm(order)
    v = similar(order)
    @inbounds @simd for i = 1:length(order)
        v[order[i]] = i
    end
    v
end