using Test, LuxurySparse, SparseArrays, LinearAlgebra

@testset "iterate" begin
    for A in Any[
        pmrand(10),
        Diagonal(randn(10)),
        IMatrix(10),
        randn(10, 10),
        sprand(10, 10, 0.5), 
    ]
        for M in Any[A, A', transpose(A)]
            O = zeros(eltype(A), size(A)...)
            for (i, j, v) in IterNz(A)
                O[i,j] = v
            end
            @test O ≈ A
        end
    end
end