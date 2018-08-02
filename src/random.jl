"""
    pmrand(T::Type, n::Int) -> PermMatrix

Return random PermMatrix.
"""
function pmrand end

pmrand(::Type{T}, n::Int) where T = PermMatrix(randperm(n), randn(T, n))
pmrand(n::Int) = pmrand(Float64, n)
