using Test
using LuxurySparse

@testset "constructors & properties" begin

# default type is Bool
@test IMatrix{2}() isa IMatrix{2, Bool}
# Vector/Matrix like constructor
@test IMatrix(2) isa IMatrix{2, Bool}

@test issparse(IMatrix(2)) == true
@test isdense(IMatrix(2)) == false

@test size(IMatrix(2)) == (2, 2)
@test size(IMatrix(2), 1) == 2
@test size(IMatrix(2), 2) == 2

@test issymmetric(IMatrix(3)) == true
@test ishermitian(IMatrix(3)) == true

@test nnz(IMatrix(3)) == 3
@test nonzeros(IMatrix(3)) == [1, 1, 1]

@test eltype(IMatrix(3)) == Bool

end
