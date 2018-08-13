var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": "CurrentModule = LuxurySparse"
},

{
    "location": "#LuxurySparse-1",
    "page": "Home",
    "title": "LuxurySparse",
    "category": "section",
    "text": "A \"Luxury\" sparse matrix library for julia language.LuxurySparse is a Julia package from QuantumBFS. It aims to provide powerful sparse matrix types for Julia, which is initially motivated by quantum simulation.It containsmore sparse matrix types like IMatrix, PermMatrix and SparseMatrixCOO,\nhigh performance type convertion, kron and multiplication operations."
},

{
    "location": "#Manual-1",
    "page": "Home",
    "title": "Manual",
    "category": "section",
    "text": "Pages = [\n    \"luxurysparse.md\",\n]\nDepth = 1"
},

{
    "location": "tutorial/#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": ""
},

{
    "location": "tutorial/#Tutorial-1",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "section",
    "text": ""
},

{
    "location": "tutorial/#Generalized-permutation-matrix-1",
    "page": "Tutorial",
    "title": "Generalized permutation matrix",
    "category": "section",
    "text": ""
},

{
    "location": "tutorial/#Example:-Control-Y-Gate-in-Quantum-simulation-1",
    "page": "Tutorial",
    "title": "Example: Control-Y Gate in Quantum simulation",
    "category": "section",
    "text": "Generalized permutation matrices are frequently used in fields such as quantum computation, group thoery. Here we see an example of Control-Y Gateleft(beginmatrix\n1  0  0  0\n0  1  0  0\n0  0  0  -i\n0  0  i  0\nendmatrixright)This data structure can be represented in the form of PermMatrixperm: [1, 2, 4, 3]\nvals: [1, 1, -i, i]Now let\'s do a benchmark to feel the speed upusing LuxurySparse: PermMatrix\npm = PermMatrix([1,2,4,3], [1,1,-im,im])v = randn(4)\n@benchmark $pm*$vBenchmarkTools.Trial: \n  memory estimate:  144 bytes\n  allocs estimate:  1\n  --------------\n  minimum time:     36.789 ns (0.00% GC)\n  median time:      38.816 ns (0.00% GC)\n  mean time:        49.227 ns (10.20% GC)\n  maximum time:     1.629 μs (89.36% GC)\n  --------------\n  samples:          10000\n  evals/sample:     992As a comparisonsp = SparseMatrixCSC(pm)\n@benchmark $sp*$vBenchmarkTools.Trial: \n  memory estimate:  144 bytes\n  allocs estimate:  1\n  --------------\n  minimum time:     64.578 ns (0.00% GC)\n  median time:      65.769 ns (0.00% GC)\n  mean time:        74.292 ns (6.80% GC)\n  maximum time:     1.419 μs (87.15% GC)\n  --------------\n  samples:          10000\n  evals/sample:     979"
},

{
    "location": "tutorial/#Identity-Matrix-1",
    "page": "Tutorial",
    "title": "Identity Matrix",
    "category": "section",
    "text": "Identity matrix is static, which is defined asstruct IMatrix{N, Tv} <: AbstractMatrix{Tv} endWith this type, the Kronecker product operation can be much faster. Now let\'s see a benchmarkusing LuxurySparse: IMatrix\nId = IMatrix{1, Float64}()\nB = randn(7,7);@benchmark kron($Id, $B)BenchmarkTools.Trial: \n  memory estimate:  0 bytes\n  allocs estimate:  0\n  --------------\n  minimum time:     1.642 ns (0.00% GC)\n  median time:      1.651 ns (0.00% GC)\n  mean time:        1.658 ns (0.00% GC)\n  maximum time:     32.101 ns (0.00% GC)\n  --------------\n  samples:          10000\n  evals/sample:     1000With the help of Julia\'s multiple dispatch, the above trivil kron operation can be avoided."
},

{
    "location": "luxurysparse/#",
    "page": "Manual",
    "title": "Manual",
    "category": "page",
    "text": ""
},

{
    "location": "luxurysparse/#LuxurySparse.IMatrix",
    "page": "Manual",
    "title": "LuxurySparse.IMatrix",
    "category": "type",
    "text": "IMatrix{N, Tv}()\nIMatrix{N}() where N = IMatrix{N, Int64}()\nIMatrix(A::AbstractMatrix{T}) where T -> IMatrix\n\nIMatrix matrix, with size N as label, use Int64 as its default type, both * and kron are optimized.\n\n\n\n\n\n"
},

{
    "location": "luxurysparse/#LuxurySparse.PermMatrix",
    "page": "Manual",
    "title": "LuxurySparse.PermMatrix",
    "category": "type",
    "text": "PermMatrix{Tv, Ti}(perm::Vector{Ti}, vals::Vector{Tv}) where {Tv, Ti<:Integer}\nPermMatrix(perm::Vector{Ti}, vals::Vector{Tv}) where {Tv, Ti}\nPermMatrix(ds::AbstractMatrix)\n\nPermMatrix represents a special kind linear operator: Permute and Multiply, which means M * v = v[perm] * val Optimizations are used to make it much faster than SparseMatrixCSC.\n\nperm is the permutation order,\nvals is the multiplication factor.\n\nGeneralized Permutation Matrix\n\n\n\n\n\n"
},

{
    "location": "luxurysparse/#LuxurySparse.SparseMatrixCOO",
    "page": "Manual",
    "title": "LuxurySparse.SparseMatrixCOO",
    "category": "type",
    "text": "SparseMatrixCOO(is::Vector, js::Vector, vs::Vector, m::Int, n::Int) -> SparseMatrixCOO\nSparseMatrixCOO{Tv, Ti}(is::Vector{Ti}, js::Vector{Ti}, vs::Vector{Tv}, m::Int, n::Int) -> SparseMatrixCOO\n\nA sparse matrix in COOrdinate format.\n\nAlso known as the ‘ijv’ or ‘triplet’ format.\n\nNotes\n\nCOO matrices should not be used in arithmetic operations like addition, subtraction, multiplication, division, and matrix power.\n\nAdvantages of the COO format\n\nfacilitates fast conversion among sparse formats\npermits duplicate entries (see example)\nvery fast conversion to and from CSR/CSC formats (CSR is not implemented)\n\nDisadvantages of the COO format\n\ndoes not directly support:\n\narithmetic operations\nslicing\n\nIntended Usage\n\nCOO is a fast format for constructing sparse matrices\nOnce a matrix has been constructed, convert to CSR or CSC format for fast arithmetic and matrix vector operations\nBy default when converting to CSR or CSC format, duplicate (i,j) entries will be summed together. This facilitates efficient construction of finite element matrices and the like. (see example)\n\n\n\n\n\n"
},

{
    "location": "luxurysparse/#LuxurySparse.allocated_coo-Union{Tuple{T}, Tuple{Type{T},Int64,Int64}} where T",
    "page": "Manual",
    "title": "LuxurySparse.allocated_coo",
    "category": "method",
    "text": "allocated_coo(::Type, N::Int, nnz::Int) -> SparseMatrixCOO\n\nConstruct a preallocated SparseMatrixCOO instance.\n\n\n\n\n\n"
},

{
    "location": "luxurysparse/#LuxurySparse.fast_invperm-Tuple{Any}",
    "page": "Manual",
    "title": "LuxurySparse.fast_invperm",
    "category": "method",
    "text": "faster invperm\n\n\n\n\n\n"
},

{
    "location": "luxurysparse/#LuxurySparse.notdense",
    "page": "Manual",
    "title": "LuxurySparse.notdense",
    "category": "function",
    "text": "notdense(M) -> Bool\n\nReturn true if a matrix is not dense.\n\nNote: It is not exactly same as isparse, e.g. Diagonal, IMatrix and PermMatrix are both notdense but not isparse.\n\n\n\n\n\n"
},

{
    "location": "luxurysparse/#LuxurySparse.pmrand",
    "page": "Manual",
    "title": "LuxurySparse.pmrand",
    "category": "function",
    "text": "pmrand(T::Type, n::Int) -> PermMatrix\n\nReturn random PermMatrix.\n\n\n\n\n\n"
},

{
    "location": "luxurysparse/#LuxurySparse.staticize",
    "page": "Manual",
    "title": "LuxurySparse.staticize",
    "category": "function",
    "text": "staticize(A::AbstractMatrix) -> AbastractMatrix\n\ntransform a matrix to a static form.\n\n\n\n\n\n"
},

{
    "location": "luxurysparse/#LuxurySparse-1",
    "page": "Manual",
    "title": "LuxurySparse",
    "category": "section",
    "text": "We provide more detailed optimization through a self-defined sparse library which is more efficient for operations related to quantum gates.Modules = [LuxurySparse]\nOrder   = [:module, :constant, :type, :macro, :function]"
},

]}
