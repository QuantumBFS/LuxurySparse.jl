using Documenter, LuxurySparse

# TODO: use Literate to process examples
# using Literate
# preprocess tutorial scripts

# make documents
makedocs(
    modules = [LuxurySparse],
    clean = false,
    format = :html,
    sitename = "LuxurySparse.jl",
    linkcheck = !("skiplinks" in ARGS),
    analytics = "UA-89508993-1",
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Manual" => "luxurysparse.md",
    ],
    html_prettyurls = !("local" in ARGS),
    html_canonical = "https://quantumbfs.github.io/LuxurySparse.jl/latest/",
)

deploydocs(
    repo = "github.com/QuantumBFS/LuxurySparse.jl.git",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
