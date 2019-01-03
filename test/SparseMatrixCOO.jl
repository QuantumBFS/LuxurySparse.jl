using Test, SparseArrays
import LuxurySparse: SparseMatrixCOO, allocated_coo, isdense

coo1 = SparseMatrixCOO([1,4,2,3,3,3], [1,1,2,4,3,4], [0.1, 0.2, 0.4im, 0.5, 0.3, 0.5im], 4, 4)

@testset "basic" begin
    @test eltype(coo1) == ComplexF64
    @test size(coo1, 1) == size(coo1, 2) == 4

    coo2 = allocated_coo(ComplexF64, 4, 4, 6)
    @test eltype(coo2) == ComplexF64
    @test nnz(coo2) == 6
    @test size(coo2) == (4, 4)
    @test size(coo2, 1) == size(coo2, 2) == 4

    @test coo1 == copy(coo1)
    @test coo1 == copyto!(coo2, coo1)

    @test Matrix(coo1) == [0.1 0 0 0; 0 0.4im 0 0; 0 0 0.3 0.5+0.5im; 0.2 0 0 0]
    @test sparse(coo1) == sparse(findnz(coo1)...)
    @test isdense(coo1) == false

    A = sprand(50,50, 0.2) |> Matrix
    @test SparseMatrixCOO(A) == A
end

@testset "sparse" begin
    p1 = copy(coo1)
    @test nonzeros(p1) == p1.vs
    p1.vs[2] = 0
    @test dropzeros!(p1) == [0.1 0 0 0; 0 0.4im 0 0; 0 0 0.3 0.5+0.5im; 0 0 0 0] 
end

@testset "basicmath" begin
    @test coo1*2 == Matrix(coo1)*2
    @test 2*coo1 == 2*Matrix(coo1)
    @test coo1/2 == Matrix(coo1)/2
    @test coo1*2 isa SparseMatrixCOO
    @test coo1/2 isa SparseMatrixCOO
    @test 2*coo1 isa SparseMatrixCOO
end
