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

model(classifier::DistributedFluxClassifier) = classifier.model.model
params(classifier::DistributedFluxClassifier) = classifier.model.params
optimizer(classifier::DistributedFluxClassifier) = classifier.model.optimiser

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


function Lighthouse.log_event!(logger::RemoteChannel, value)
    logged = string(now(), " | ", value)
    put!(logger, "events" => logged)
    return logged
end

function toarray(gs::Zygote.Grads)
    n = sum(length(p) for p in gs.params)
    a = Array{eltype(first(gs.params))}(undef, n)
    copy!(gs, a)
    return a
end

function loss_and_gradient(classifier::FluxClassifier, weights, batchin, logger)
    batch = LighthouseFlux.assemble_batch(batchin)
    @info "batch of type $(typeof(batch))"
    @info "logger of type $(typeof(logger))"
    train_loss, back = log_resource_info!(logger, "train/forward_pass";
                                          suffix="_per_batch") do
        f = () -> loss(classifier.model, batch...)
        return Zygote.pullback(f, weights)
        # return Zygote.pullback(f, LighthouseFlux.params(classifier))
        # return Zygote.pullback(f, Zygote.Params(LighthouseFlux.params(classifier)))
    end
    log_value!(logger, "train/loss_per_batch", train_loss)
    gradients = log_resource_info!(logger, "train/reverse_pass";
                                   suffix="_per_batch") do
        return back(Zygote.sensitivity(train_loss))
    end
    return (train_loss, toarray(gradients))
end

function loss_and_gradient(classifier::DistributedFluxClassifier, weights, batch, logger)
    shards = shard(classifier.pids, batch...)
    #shards = Dict{Int,Any}( p => (loss_and_gradient, classifier.model, b, logger) for (p,b) in shards)
    shards = Dict{Int,Any}( p => (loss_and_gradient, classifier.model, weights, b, logger) for (p,b) in shards)
    return_channel = remotecall_fetch_all(shards)
    train_loss = 0.0
    gradients = zeros(eltype(first(weights)), sum(length(p) for p in weights))
    @info "iterating through return_channel"
    for _ in shards
        (p, (loss, grad)) = take!(return_channel)
        @show p
        @show loss
        train_loss += loss
        gradients += grad
    end
    return train_loss, copy_gradients(weights, gradients)
end

function copy_gradients(ps::Zygote.Params, ga::AbstractVector)
    gs = Zygote.Grads(IdDict{Any,Any}(), ps)
    for p in ps
        gs.grads[p] = p
    end
    copy!(gs, ga)
    return gs
end

function Lighthouse.train!(classifier::AbstractFluxClassifier, batches, logger)
    Flux.trainmode!(LighthouseFlux.model(classifier))
    weights = Zygote.Params(LighthouseFlux.params(classifier))
    @info "Starting train! loop"
    for batch in batches
        #train_loss, gradients = loss_and_gradient(classifier, batch, logger)
        train_loss, gradients = loss_and_gradient(classifier, weights, batch, logger)
        log_resource_info!(logger, "train/update"; suffix="_per_batch") do
            Flux.Optimise.update!(optimizer(classifier), weights, gradients)
            return nothing
        end
    end
    Flux.testmode!(LighthouseFlux.model(classifier))
    return nothing
end

function Lighthouse.loss_and_prediction(classifier::FluxClassifier, batch...)
    return Flux.cpu(loss_and_prediction(classifier.model, batch...))
end

