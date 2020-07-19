include("distributed/sharding.jl")
include("distributed/logging.jl")
# everything in `distributed/` could be broken off into its own package
# it is not specific to Flux or Lighthouse

struct DistributedFluxClassifier <: AbstractFluxClassifier
    pids::AbstractVector{Int}
    model::FluxClassifier
end

Lighthouse.classes(classifier::DistributedFluxClassifier) = classifier.model.classes

function Lighthouse.is_early_stopping_exception(::DistributedFluxClassifier, exception)
    return exception isa Flux.Optimise.StopException
end

Lighthouse.onehot(classifier::DistributedFluxClassifier, label) = classifier.model.onehot(label)

Lighthouse.onecold(classifier::DistributedFluxClassifier, label) = classifier.model.onecold(label)

LighthouseFlux.model(classifier::DistributedFluxClassifier) = classifier.model.model
LighthouseFlux.params(classifier::DistributedFluxClassifier) = classifier.model.params
LighthouseFlux.optimizer(classifier::DistributedFluxClassifier) = classifier.model.optimiser

"""
    loss_and_prediction(model, input_batch, other_batch_arguments...)

Return `(model_loss, model_prediction)` where:

- `model_loss` is equivalent to (and defaults to) `loss(model, input_batch, other_batch_arguments...)`.

- `model_prediction` is a matrix where the `i`th column is the soft label prediction for the `i`th
sample in `input_batch`. Thus, the numnber of columns should be `size(input_batch)[end]`, while the
number of rows is equal to the number of possible classes predicted by model. `model_prediction`
defaults to `model(input_batch)`.

This method must be implemented for all `model`s passed to [`FluxClassifier`](@ref), but
has the default return values described above, so it only needs to be overloaded if the
default definitions do not yield the expected values for a given `model` type. It
additionally may be overloaded to avoid redundant computation if `model`'s loss
function computes soft labels as an intermediate result.
"""
function loss_and_prediction(model::DistributedFluxClassifier, input_batch, other_batch_arguments...)
    return (loss(model, input_batch, other_batch_arguments...), model(input_batch))
end


# SETUP
"""
    setup_assemble_batch_s3(model::DistributedFluxClassifier)

Sets up all workers with a `LighthouseFlux.assemble_batch()` function that reads from s3.
"""
function setup_assemble_batch_s3(pids::AbstractVector{Int})
    
end

function loss_and_gradient(classifier::DistributedFluxClassifier, batch, logger)
    shards = shard(classifier.pids, batch...)
    shards = Dict{Int,Any}( p => (loss_and_gradient, classifier.model, b, logger) for (p,b) in shards)
    #shards = Dict{Int,Any}( p => (loss_and_gradient, classifier.model, weights, b, logger) for (p,b) in shards)
    return_channel = remotecall_fetch_all(shards)
    train_loss = 0
    gradients = 0
    @info "iterating through return_channel"
    for _ in shards
        (p, (loss, grad)) = take!(return_channel)
        @show p
        @show loss
        @show size(grad)
        train_loss += loss
        gradients += grad
    end
    return train_loss, gradients
end

function Lighthouse.loss_and_prediction(classifier::FluxClassifier, batch...)
    return Flux.cpu(loss_and_prediction(classifier.model, batch...))
end

# function shard(ps::AbstractVector{Int}, xs...)
#     xs = collect(xs)
#     # sort both for shard consistency
#     sort!(xs)
#     sort!(ps)
#     return Dict{Int,Any}(p => part for (part, p) in zip(Iterators.partition(xs, length(xs) ÷ length(ps)), ps))
# end
# 
# function remotecall_fetch_all(objects::Dict{Int,T}) where T
#     channel = Channel(length(objects))
#     for (p, obj) in objects
#         @async begin
#             result = remotecall_fetch(p, obj...)
#             put!(channel, p => result)
#         end
#     end
#     return channel
# end

