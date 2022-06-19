using Test, LuxurySparse, SparseArrays, LinearAlgebra

@testset "iterate" begin
    for M in Any[
        pmrand(10),
        Diagonal(randn(10)),
        IMatrix(10),
        randn(10, 10),
        sprand(10, 10, 0.5), 
    ]
        for A in Any[M, M', transpose(M)]
            O = zeros(eltype(A), size(A)...)
            @test eltype(IterNz(A)) == eltype(A)
            for (i, j, v) in it
                O[i,j] = v
            end
            @test O ≈ A
        end
    end
    # static
    a = sparse([1,2,3], [3,4,4], randn(3), 4, 4)
    A = LuxurySparse.staticize(a)
    O = zeros(eltype(A), size(A)...)
    for (i, j, v) in IterNz(A)
        O[i,j] = v
    end
    @test O ≈ a
end