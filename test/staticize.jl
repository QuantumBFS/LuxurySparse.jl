using Test
using LinearAlgebra
using Random

using LuxurySparse
import LuxurySparse: staticize
using StaticArrays: SVector, SMatrix

Random.seed!(2)

using Compat.Test
@testset "staticize" begin
    # permmatrix
    m = pmrand(ComplexF64, 4)
    sm = m |> staticize
    @test sm isa SPermMatrix{4, ComplexF64}
    @test sm.perm isa SVector
    @test sm.vals isa SVector
    @test sm.perm == m.perm
    @test sm.vals == m.vals

    # csc
    m = sprand(ComplexF64, 4,4, 0.5)
    sm = m |> staticize
    @test sm.colptr isa SVector
    @test sm.rowval isa SVector
    @test sm.nzval isa SVector
    @test sm.nzval == m.nzval
    @test sm.rowval == m.rowval
    @test sm.colptr == m.colptr

    # diagonal
    m = Diagonal(randn(ComplexF64, 4))
    sm = m |> staticize
    @test sm.diag isa SVector
    @test sm.diag == m.diag

    # dense vector
    m = randn(ComplexF64, 4)
    sm = m |> staticize
    @test sm isa SVector
    @test sm == m

    # dense matrix
    m = randn(ComplexF64, 4, 4)
    sm = m |> staticize
    @test sm isa SMatrix
    @test sm == m
end
