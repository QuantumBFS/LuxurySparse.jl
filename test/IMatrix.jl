using Test, Random
using LuxurySparse
using SparseArrays
using LinearAlgebra

Random.seed!(2)

p1 = IMatrix(4)
sp = sprand(ComplexF64, 4, 4, 0.5)
ds = rand(ComplexF64, 4, 4)
pm = PermMatrix([2, 3, 4, 1], randn(4))
v = [0.5, 0.3im, 0.2, 1.0]
dv = Diagonal(v)

@testset "basic" begin
    @test p1 == copy(p1)
    @test eltype(p1) == Bool
    @test size(p1) == (4, 4)
    @test size(p1, 1) == size(p1, 2) == 4
    @test Matrix(p1) == [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

    p0 = similar(p1, ComplexF64)
    @test p0 !== p1
    p2 = copyto!(p1, p0)
    @test p2 == p0
    @test_throws DimensionMismatch copyto!(p0, IMatrix(2))
end

@testset "conversion" begin
    for mat in Any[p1, pm, dv]
        @test mat == SparseMatrixCSC(mat)
        @test mat == Matrix(mat)
    end
    for mat in Any[p1, pm, dv]
        @test mat == PermMatrix(mat)
    end
    @test Diagonal(p1) == p1

    @test SparseMatrixCSC(Diagonal(fill(2, 4))) ≈ Diagonal(fill(2, 4))
end

@testset "sparse" begin
    @show p1
    @test LuxurySparse.length(IterNz(p1)) == 4
end

@testset "linalg" begin
    for op in Any[conj, real, transpose, copy, inv]
        @test op(p1) == Matrix(I, 4, 4)
        @test typeof(op(p1)) == typeof(p1)
    end
    @test imag(p1) == zeros(4, 4)
    @test p1' == Matrix(I, 4, 4)
    @test ishermitian(p1)
end

@testset "elementary" begin
    @test all(isapprox.(conj(p1), conj(Matrix(p1))))
    @test all(isapprox.(real(p1), real(Matrix(p1))))
    @test all(isapprox.(imag(p1), imag(Matrix(p1))))
end

@testset "basicmath" begin
    @test p1 * 2im == Matrix(p1) * 2im
    @test p1 / 2.0 == Matrix(p1) / 2.0
end

@testset "push coverage" begin
    @test SparseMatrixCSC(IMatrix(3)) ≈ Diagonal(ones(3))
    @test Diagonal(IMatrix(3)) ≈ Diagonal(ones(3))
    @test IMatrix(Diagonal(ones(3))) === IMatrix{Float64}(3)
    @test IMatrix{ComplexF64}(Diagonal(ones(3))) === IMatrix{ComplexF64}(3)
    @test_throws DimensionMismatch IMatrix(ones(3, 5))
    @test_throws DimensionMismatch IMatrix{ComplexF64}(ones(3, 5))
    @test IMatrix(3) .* 2 == IMatrix(3) * 2 == Diagonal([2,2,2])
    @test 2 .* IMatrix(3) == 2 * IMatrix(3) == Diagonal([2,2,2])
    @test IMatrix(3) ≈ IMatrix{Float64}(3)
end