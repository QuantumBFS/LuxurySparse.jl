using Test
using LinearAlgebra, SparseArrays, Random
using LuxurySparse

Random.seed!(2)

p1 = IMatrix{4}()
sp = sprand(ComplexF64, 4, 4, 0.5)
ds = rand(ComplexF64, 4, 4)
pm = PermMatrix([2, 3, 4, 1], randn(4))
v = [0.5, 0.3im, 0.2, 1.0]
dv = Diagonal(v)

@testset "invdet" begin
    ####### linear algebra  ######
    @test inv(p1) == inv(Matrix(p1))
    @test det(p1) == det(Matrix(p1))
    @test diag(p1) == diag(Matrix(p1))
    @test logdet(p1) == 0
    @test inv(pm) == inv(Matrix(pm))

    for m in [pm, sp, p1, dv]
        @test !(m |> isdense)
        @test !(m' |> isdense)
        @test !(transpose(m) |> isdense)
    end
    for m in [ds, v]
        @test m |> isdense
        @test m' |> isdense
        @test transpose(m) |> isdense
    end
end

@testset "multiply" begin
    for source_ in [p1, sp, ds, dv, pm]
        for target in [p1, sp, ds, dv, pm]
            for source in [source_, source_', transpose(source_)]
                lres = source * target
                rres = target * source
                flres = Matrix(source) * Matrix(target)
                frres = Matrix(target) * Matrix(source)
                @test lres ≈ flres
                @test rres ≈ frres
                if !(target === p1 || parent(source) === p1)
                    @test eltype(lres) == eltype(flres)
                    @test eltype(rres) == eltype(frres)
                end
                if !(target |> isdense) && !(parent(source) |> isdense)
                    @test lres |> isdense == false
                    @test lres |> isdense == false
                end
            end
        end
    end
end

@testset "mul-vector" begin
    # permutation multiply
    lres = transpose(conj(v)) * pm  #! v' not realized!
    rres = pm * v
    flres = v' * Matrix(pm)
    frres = Matrix(pm) * v
    @test lres == flres
    @test rres == frres
    @test eltype(lres) == eltype(flres)
    @test eltype(rres) == eltype(frres)

    # IMatrix
    @test v' * p1 == v'
    @test p1 * v == v
end

@testset "sparse-diag" begin
    N = 100
    dg = Diagonal(randn(ComplexF64, 1000))
    sp = SparseMatrixCSC(dg)
    @test sp * dg == sp * sp
    @test dg * sp == sp * sp
end


@testset "randn" begin
    Random.seed!(2)
    T = ComplexF64
    for m in [sprand(T, 5, 5, 0.5)]
        zm = zero(m)
        @test zm ≈ zeros(T, 5, 5)
        if VERSION < v"1.4.0"
            rand!(zm)
            @test !(zm ≈ zeros(T, 5, 5))
            zm = zero(m)
            randn!(zm)
            @test !(zm ≈ zeros(T, 5, 5))
        end
    end
    for m in [pmrand(T, 5), Diagonal(randn(T, 5))]
        zm = zero(m)
        @test zm ≈ zeros(T, 5, 5)
        rand!(zm)
        @test !(zm ≈ zeros(T, 5, 5))
        zm = zero(m)
        randn!(zm)
        @test !(zm ≈ zeros(T, 5, 5))
    end
end

@testset "multiply rectangular matrix" begin
    pm = pmrand(4)
    sp = sparse(reshape([1.0 2 3 4], 4, 1))
    res = pm * sp
    @test res ≈ Matrix(pm) * Matrix(sp)
end

@testset "findnz" begin
    for m in [p1, sp, ds, dv, pm]
        for _m in [m, staticize(m)]
            out = zeros(eltype(m), size(m)...)
            for (i, j, v) in zip(LuxurySparse.findnz(_m)...)
                out[i, j] = v
            end
            @test out ≈ m
        end
    end
end

@testset "fallback-issue#Yao#127" begin
    pm = pmrand(512)
    sp = sprand(512, 512, 0.05)
    @test nnz(pm * sp - SparseMatrixCSC(pm) * sp) == 0
end

@testset "regression test" begin
    pop = sparse([2, 3], [4, 5], [1, √2], 5, 5)
    zop = Diagonal([0, 1 / 2, 1, -1 / 2, 0])
    @test (pop*zop)[2, 4] == -0.5
end
