export SDPermMatrix,
    SPermMatrix,
    SDSparseMatrixCSC,
    SDMatrix,
    SDDiagonal,
    SDVector,
    staticize,
    dynamicize

######### Union of static and dynamic matrices ##########
const SDPermMatrix = PermMatrix
const SPermMatrix{N, Tv, Ti} = PermMatrix{Tv, Ti, <:SVector{N, Tv}, <:SVector{N, Ti}}
const SDSparseMatrixCSC = Union{SparseMatrixCSC, SSparseMatrixCSC}
const SDMatrix = Union{Matrix, SMatrix}
const SDDiagonal = Union{Diagonal, SDiagonal}
const SDVector = Union{Vector, SVector}

######### staticize ##########
"""
    staticize(A::AbstractMatrix) -> AbastractMatrix

transform a matrix to a static form.
"""
function staticize end

staticize(A::AbstractMatrix) = SMatrix{size(A,1), size(A,2)}(A)
staticize(A::AbstractVector) = SVector{length(A)}(A)
staticize(A::Diagonal) = SDiagonal{size(A,1)}((A.diag...,))
staticize(A::PermMatrix) = PermMatrix(SVector{size(A,1)}(A.perm), SVector{size(A, 1)}(A.vals))
function staticize(A::SparseMatrixCSC)
    SSparseMatrixCSC(A.m, A.n, SVector{length(A.colptr)}(A.colptr), SVector{length(A.rowval)}(A.rowval), SVector{length(A.nzval)}(A.nzval))
end

"""
    dynamicize(A::AbstractMatrix) -> AbastractMatrix

transform a matrix to a dynamic form.
"""
function dynamicize end

dynamicize(A::SMatrix) = Matrix(A)
dynamicize(A::SVector) = Vector(A)
dynamicize(A::SDiagonal) = Diagonal(Vector(A.diag))
dynamicize(A::PermMatrix) = PermMatrix(Vector(A.perm), Vector(A.vals))
function dynamicize(A::SSparseMatrixCSC)
    SparseMatrixCSC(A.m, A.n, Vector(A.colptr), Vector(A.rowval), Vector(A.nzval))
end
