# LighthouseFlux.jl

[![CI](https://github.com/beacon-biosignals/LighthouseFlux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/beacon-biosignals/LighthouseFlux.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/beacon-biosignals/LighthouseFlux.jl/branch/main/graph/badge.svg?token=aOni8ATb88)](https://codecov.io/gh/beacon-biosignals/LighthouseFlux.jl)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://beacon-biosignals.github.io/LighthouseFlux.jl/stable)
[![Docs: development](https://img.shields.io/badge/docs-dev-blue.svg)](https://beacon-biosignals.github.io/LighthouseFlux.jl/dev)


LighthouseFlux provides a `FluxClassifier` wrapper that implements the [Lighthouse](https://github.com/beacon-biosignals/Lighthouse.jl) `AbstractClassifier` interface, thus enabling Flux models to easily utilize Lighthouse's training/testing harness. Assuming your model obeys normal Flux model conventions, hooking it up to LighthouseFlux generally only requires a single method overload (`LighthouseFlux.loss`).

See this package's tests for example usage.

## Installation

To install LighthouseFlux for development, run:

```
julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/beacon-biosignals/LighthouseFlux.jl"))'
```

This will install LighthouseFlux to the default package development directory, `~/.julia/dev/LighthouseFlux`.
