# LighthouseFlux.jl

[![Build Status](https://travis-ci.com/beacon-biosignals/LighthouseFlux.jl.svg?token=Jbjm3zfgVHsfbKqsz3ki&branch=master)](https://travis-ci.com/beacon-biosignals/LighthouseFlux.jl)
[![codecov](https://codecov.io/gh/beacon-biosignals/LighthouseFlux.jl/branch/master/graph/badge.svg?token=aOni8ATb88)](https://codecov.io/gh/beacon-biosignals/LighthouseFlux.jl)

LighthouseFlux provides a `FluxClassifier` wrapper that implements the [Lighthouse](https://github.com/beacon-biosignals/Lighthouse.jl) `AbstractClassifier` interface, thus enabling Flux models to easily utilize Lighthouse's training/testing harness. Assuming your model obeys normal Flux model conventions, hooking it up to LighthouseFlux generally only requires a single method overload (`LighthouseFlux.loss`).

See this package's tests for example usage.

## Installation

To install LighthouseFlux for development, run:

```
julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/beacon-biosignals/LighthouseFlux.jl"))'
```

This will install LighthouseFlux to the default package development directory, `~/.julia/dev/LighthouseFlux`.

## Viewing Documentation

LighthouseFlux is currently in stealth mode (shhh!) so we're not hosting the documentation publicly yet. To view LighthouseFlux's documentation, simply build it locally and view the generated HTML in your browser:

```sh
./LighthouseFlux/docs/build.sh
open LighthouseFlux/docs/build/index.html # or whatever command you use to open HTML
```
