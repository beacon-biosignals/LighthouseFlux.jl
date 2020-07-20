module LighthouseFlux


using Dates: now
using Zygote: Zygote
using Flux: Flux
using Lighthouse: Lighthouse, classes, log_resource_info!, log_value!

using Distributed, BSON
using FastS3
using Serialization

abstract type AbstractFluxClassifier <: Lighthouse.AbstractClassifier end

include("local.jl")
include("distributed.jl")

export FluxClassifier, DistributedLogger, DistributedFluxClassifier

# stuff that is defined in `distributed/*` should live in its own package outside of LighthouseFlux
export defineat

end # module
