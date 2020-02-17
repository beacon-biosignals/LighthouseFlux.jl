using LighthouseFlux
using Documenter

makedocs(modules=[LighthouseFlux],
         sitename="LighthouseFlux",
         authors="Beacon Biosignals and other contributors",
         pages=["API Documentation" => "index.md"])

# this is commented out until we figure out how to do this privately
# deploydocs(repo="github.com/beacon-biosignals/LighthouseFlux.jl.git")
