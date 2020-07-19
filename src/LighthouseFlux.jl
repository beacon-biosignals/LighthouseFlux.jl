module LighthouseFlux

using Dates: now
using Zygote: Zygote
using Flux: Flux
using Lighthouse: Lighthouse, classes, log_resource_info!, log_value!

using Distributed, BSON

abstract type AbstractFluxClassifier <: Lighthouse.AbstractClassifier end

include("local.jl")
include("distributed.jl")

export FluxClassifier, DistributedLogger, DistributedFluxClassifier

end # module
