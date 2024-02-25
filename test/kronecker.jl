using Test, Random, SparseArrays, LinearAlgebra
import LuxurySparse: IMatrix, PermMatrix, PermMatrixCSC, basetype, AbstractPermMatrix

@testset "kron" begin
    Random.seed!(2)

    p1 = IMatrix(4)
    sp = sprand(ComplexF64, 4, 4, 0.5)
    ds = rand(ComplexF64, 4, 4)
    pm = PermMatrix([2, 3, 4, 1], randn(4))
    pmc = PermMatrixCSC([2, 3, 4, 1], randn(4))
    v = [0.5, 0.3im, 0.2, 1.0]
    dv = Diagonal(v)

    for source in Any[p1, sp, ds, dv, pm, pmc],
            target in Any[p1, sp, ds, dv, pm, pmc]
        if source isa AbstractPermMatrix && target isa AbstractPermMatrix && basetype(source) != basetype(target)
            continue
        end
        lres = kron(source, target)
        rres = kron(target, source)
        flres = kron(Matrix(source), Matrix(target))
        frres = kron(Matrix(target), Matrix(source))
        @test lres == flres
        @test rres == frres
        @test eltype(lres) == eltype(flres)
        @test eltype(rres) == eltype(frres)
        if !(target === ds && source === ds)
            @test !(typeof(lres) <: StridedMatrix)
            @test !(typeof(rres) <: StridedMatrix)
        end
    end
end