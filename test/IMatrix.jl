using Test, LinearAlgebra
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

@testset "matmul" begin

IdA = IMatrix(3)
B = rand(3)

for each in [B, transpose(B), adjoint(B), Diagonal(B), Hermitian(B), Symmetric(B), Tridiagonal(B)]
    @test IdA * each == each
    @test IdA * each === each
    @test each * IdA == each
    @test IdA * each === each
end

IdB = IMatrix{3, Float64}()

@test IdA * IdB == IdB
@test IdA * IdB === IdB

@test IdA * IdA == IdA
@test IdA * IdA === IdA

v = rand(3)

@test IdA * v == v
@test IdA * v === v

end
