using LighthouseFlux
using Documenter

makedocs(; modules=[LighthouseFlux], sitename="LighthouseFlux",
         authors="Beacon Biosignals and other contributors",
         pages=["API Documentation" => "index.md"])

deploydocs(; repo="github.com/beacon-biosignals/LighthouseFlux.jl.git", devbranch="master")
