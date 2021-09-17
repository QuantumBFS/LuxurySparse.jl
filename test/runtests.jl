using Test
using LinearAlgebra
using SparseArrays
using LuxurySparse
using Aqua
Aqua.test_all(LuxurySparse)

@testset "IMatrix" begin
    include("IMatrix.jl")
end

@testset "PermMatrix" begin
    include("PermMatrix.jl")
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
