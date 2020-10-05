# this should be upstreamed to Flux.jl/src/optimise/optimisers.jl

using LinearAlgebra

mutable struct Renormalize
    m::Float64
    M:: Float64
end

Renormalize(;m=0.1, M=1000.0) = Renormalize(m, M)

function Flux.Optimise.apply!(o::Renormalize, x, Δ)
    @. Δ = Δ * max(o.m, min(o.M, norm(x))) / norm(Δ)
    return Δ
end
