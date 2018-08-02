using Test, SparseArrays
using LuxurySparse

p1 = PermMatrix([1, 4, 2, 3], ComplexF64[0.1, 0.2, 0.4im, 0.5])
p2 = PermMatrix([2, 1, 4, 3], Float64[0.1, 0.2, 0.4, 0.5])
p3 = pmrand(4)
sp = sprand(4, 4, 0.3)
v = [0.5, 0.3im, 0.2, 1.0]


@testset "constructor" begin

@test p1          == copy(p1)
@test p1         !== copy(p1)
@test eltype(p1)  == ComplexF64
@test eltype(p2)  == Float64
@test eltype(p3)  == Float64
@test size(p1)    == (4, 4)
@test size(p3)    == (4, 4)
@test size(p1, 1) == size(p1, 2) == 4

end

@testset "conversion" begin
end
