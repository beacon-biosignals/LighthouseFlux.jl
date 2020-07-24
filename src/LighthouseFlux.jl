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

# everything in `distributed/` should go somewhere else, it has nothing to do with Lighthouse of Flux
include("distributed/dataloader.jl")
include("distributed/logging.jl")
include("distributed/sharding.jl")

include("distributed.jl")

export FluxClassifier, DistributedLogger, DistributedFluxClassifier

export @defineat, buffered_batch_loader

end # module
