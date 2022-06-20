using Test, LuxurySparse

@testset "basic" begin
    m = sprand(ComplexF64, 4, 4, 0.5)
    sm = m |> staticize
    @test Matrix(m) == Matrix(sm)
    allsame = true
    for i=1:4, j=1:4
        allsame &= m[i,j] == sm[i,j]
    end
    @test allsame
    @test LuxurySparse.nnz(sm) == LuxurySparse.nnz(m)
    @test LuxurySparse.nonzeros(sm) == LuxurySparse.nonzeros(m)
    @test issparse(sm)
    @test LuxurySparse.dropzeros!(sm) == sm
    @test sm == m
end