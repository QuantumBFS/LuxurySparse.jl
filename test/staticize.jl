using Test
using LinearAlgebra, SparseArrays, Random

using LuxurySparse
import LuxurySparse: staticize
using StaticArrays: SVector, SMatrix

Random.seed!(2)

@testset "staticize" begin
    # permmatrix
    m = pmrand(ComplexF64, 4)
    sm = m |> staticize
    @test sm isa SPermMatrix{4, ComplexF64}
    @test sm.perm isa SVector
    @test sm.vals isa SVector
    @test sm.perm == m.perm
    @test sm.vals == m.vals
    dm = sm |> dynamicize
    @test dm isa PermMatrix{ComplexF64}
    @test dm.perm isa Vector
    @test dm.vals isa Vector
    @test dm.perm == m.perm
    @test dm.vals == m.vals

    # csc
    m = sprand(ComplexF64, 4,4, 0.5)
    sm = m |> staticize
    @test sm.colptr isa SVector
    @test sm.rowval isa SVector
    @test sm.nzval isa SVector
    @test sm.nzval == m.nzval
    @test sm.rowval == m.rowval
    @test sm.colptr == m.colptr

    dm = sm |> dynamicize
    @test dm.colptr isa Vector
    @test dm.rowval isa Vector
    @test dm.nzval isa Vector
    @test dm.nzval == m.nzval
    @test dm.rowval == m.rowval
    @test dm.colptr == m.colptr

    # diagonal
    m = Diagonal(randn(ComplexF64, 4))
    sm = m |> staticize
    @test sm.diag isa SVector
    @test sm.diag == m.diag
    dm = sm |> dynamicize
    @test dm.diag isa Vector
    @test dm.diag == m.diag

    # dense vector
    m = randn(ComplexF64, 4)
    sm = m |> staticize
    @test sm isa SVector
    @test sm == m
    dm = sm |> dynamicize
    @test dm isa Vector
    @test dm == m

    # dense matrix
    m = randn(ComplexF64, 4, 4)
    sm = m |> staticize
    @test sm isa SMatrix
    @test sm == m
    dm = sm |> dynamicize
    @test dm isa Matrix
    @test dm == m
end

@testset "SSparseMatrixCSC" begin
    m = sprand(ComplexF64, 4,4, 0.5)
    sm = m |> staticize
    @test nnz(sm) == nnz(m)
    @test nonzeros(sm) == nonzeros(m)
    @test isdense(sm) == false
    @test issparse(sm)
    @test dropzeros!(sm) == sm
    @test sm == m
end
