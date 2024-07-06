# Generate documentation with this command:
# (cd docs && julia make.jl)

push!(LOAD_PATH, "..")

using Documenter
using ShortFFTs

makedocs(; sitename="ShortFFTs", format=Documenter.HTML(), modules=[ShortFFTs])

deploydocs(; repo="github.com/eschnett/ShortFFTs.jl.git", devbranch="main", push_preview=true)
