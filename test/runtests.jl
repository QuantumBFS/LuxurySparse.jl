using Test
using Aqua
using LuxurySparse

@testset "Aqua" begin
    Aqua.test_all(LuxurySparse)
end

@testset "IMatrix" begin
    include("IMatrix.jl")
end

@testset "PermMatrix" begin
    include("PermMatrix.jl")
    include("PermMatrixCSC.jl")
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
