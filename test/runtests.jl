using Test
using LinearAlgebra, SparseArrays
using LuxurySparse

@testset "IMatrix" begin
    include("IMatrix.jl")
end

@testset "PermMatrix" begin
    include("PermMatrix.jl")
end

@testset "SparseMatrixCOO" begin
    include("SparseMatrixCOO.jl")
end

@testset "SSparseMatrixCSC" begin
    include("SSparseMatrixCSC.jl")
end

@testset "kronecker" begin
    include("kronecker.jl")
end

@testset "linalg" begin
    include("linalg.jl")
end

@testset "staticize" begin
    include("staticize.jl")
end

@testset "broadcast" begin
    include("broadcast.jl")
end

@testset "iterate" begin
    include("iterate.jl")
end
