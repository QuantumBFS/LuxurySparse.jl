using Test
using LuxurySparse
using LinearAlgebra
using SparseArrays

@testset "broadcast *" begin

    @testset "Diagonal .* $(nameof(typeof(M)))" for M in [[pmrand(3)]..., pmcscrand(3)]
        M1 = Diagonal(rand(3))
        out = M1 .* M
        @test typeof(out) <: Diagonal
        @test out ≈ Matrix(M1) .* M

        out = M .* M1
        @test typeof(out) <: Diagonal
        @test out ≈ Matrix(M) .* M1
    end

    @testset "PermMatrix .* $(nameof(typeof(M)))" for M in Any[
        rand(3, 3),
        pmrand(3),
        sprand(3, 3, 0.5),
    ]
        M1 = pmrand(3)
        out = M1 .* M
        @test typeof(out) <: PermMatrix
        @test out ≈ Matrix(M1) .* M

        out = M .* M1
        @test typeof(out) <: PermMatrix
        @test out ≈ M .* Matrix(M1)

        M1 = pmcscrand(3)
        out = M1 .* M
        @test typeof(out) <: PermMatrixCSC
        @test out ≈ Matrix(M1) .* M

        out = M .* M1
        !(M isa PermMatrix) && @test typeof(out) <: PermMatrixCSC
        @test out ≈ M .* Matrix(M1)
    end

    @testset "IMatrix .* $(nameof(typeof(M)))" for M in Any[
        rand(3, 3),
        pmrand(3),
        pmcscrand(3),
        sprand(3, 3, 0.5),
    ]
        eye = IMatrix(3)
        out = eye .* M
        @test typeof(out) <: Diagonal
        @test out ≈ Matrix(eye) .* M

        out = M .* eye
        @test typeof(out) <: Diagonal
        @test out ≈ M .* Matrix(eye)
    end

    @test IMatrix(3) .* IMatrix(3) === IMatrix(3)
    d = Diagonal(rand(3))
    sp = sprand(3, 3, 0.5)
    @test d .* sp ≈ Matrix(d) .* Matrix(sp)
    @test typeof(d .* sp) <: Diagonal

    @testset "Number .* $(nameof(typeof(M)))" for M in Any[pmrand(3), Diagonal(randn(3))]
        @test 3.0 .* M ≈ 3.0 .* Matrix(M) && M .* 3.0 ≈ Matrix(M) .* 3.0
        @test typeof(M .* 3.0) == typeof(M)
    end
end


@testset "broadcast -" begin

    @testset "Diagonal .- $(nameof(typeof(M)))" for M in Any[pmrand(3)]
        M1 = Diagonal(rand(3))
        @test M1 .- M ≈ Matrix(M1) .- M
        @test M .- M1 ≈ Matrix(M) .- M1
    end

    @testset "PermMatrix .* $(nameof(typeof(M)))" for M in Any[
        2.0,
        rand(3, 3),
        pmrand(3),
        sprand(3, 0.5),
        sprand(3, 3, 0.5),
    ]
        M1 = pmrand(3)
        @test M1 .- M ≈ Matrix(M1) .- M
        @test M .- M1 ≈ M .- Matrix(M1)

        M1 = pmcscrand(3)
        @test M1 .- M ≈ Matrix(M1) .- M
        @test M .- M1 ≈ M .- Matrix(M1)
    end

    @testset "IMatrix .* $(nameof(typeof(M)))" for M in Any[
        2.0,
        rand(3, 3),
        pmrand(3),
        sprand(3, 0.5),
        sprand(3, 3, 0.5),
    ]
        eye = IMatrix(3)
        @test eye .- M ≈ Matrix(eye) .- M

        @test M .- eye ≈ M .- Matrix(eye)
    end

    @test IMatrix(3) .- IMatrix(3) ≈ zeros(3, 3)
end
