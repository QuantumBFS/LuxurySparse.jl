using Test, LinearAlgebra, Random
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

srand(2)

p1 = IMatrix{4}()
sp = sprand(ComplexF64, 4,4, 0.5)
ds = rand(ComplexF64, 4,4)
pm = PermMatrix([2,3,4,1], randn(4))
v = [0.5, 0.3im, 0.2, 1.0]
dv = Diagonal(v)

@testset "basic" begin
    @test p1==copy(p1)
    @test eltype(p1) == Bool
    @test size(p1) == (4, 4)
    @test size(p1, 1) == size(p1, 2) == 4
    @test Matrix(p1) == [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
end

@testset "conversion" begin
    for mat in [p1, pm, dv]
        @test mat == SparseMatrixCSC(mat)
        @test mat == Matrix(mat)
        @test mat == typeof(mat)(Matrix(mat))
        @test mat == typeof(mat)(SparseMatrixCSC(mat))
    end
    for mat in [p1, pm, dv]
        @test mat == PermMatrix(mat)
        @test mat == typeof(mat)(PermMatrix(mat))
    end
    @test Diagonal(p1) == p1
    @test p1 == typeof(p1)(Diagonal(p1))
end

@testset "sparse" begin
    @test nnz(p1) == 4
    @test nonzeros(p1) == ones(4)
end

@testset "linalg" begin
    for op in [conj, real, transpose, copy, inv]
        @test op(p1) == Matrix(I, 4, 4)
        @test typeof(op(p1)) == typeof(p1)
    end
    @test imag(p1) == zeros(4, 4)
    @test p1' == Matrix(I, 4, 4)

    # This will be lazy evaluated in 0.7+
    @static if VERSION < v"0.7-"
        @test typeof(p1') == typeof(p1)
    end

    @test ishermitian(p1)
end

@testset "elementary" begin
    @test all(isapprox.(conj(p1), conj(Matrix(p1))))
    @test all(isapprox.(real(p1), real(Matrix(p1))))
    @test all(isapprox.(imag(p1), imag(Matrix(p1))))
end

@testset "basicmath" begin
    @test p1*2im == Matrix(p1)*2im
    @test p1/2.0 == Matrix(p1)/2.0
end
