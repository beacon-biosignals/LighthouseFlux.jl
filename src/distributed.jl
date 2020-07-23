include("distributed/sharding.jl")
include("distributed/logging.jl")
include("distributed/dataloader.jl")
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

function loss_and_gradient(classifier::FluxClassifier, logger::RemoteChannel)
    batch = try
        take!(_batches)
    catch e
        return nothing
    end
    # @info "batch of type $(typeof(batch))"
    # @info "logger of type $(typeof(logger))"
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

function loss_and_gradient(classifier::DistributedFluxClassifier, weights, b, logger::RemoteChannel)
    shards = Dict{Int,Any}( p => (loss_and_gradient, classifier.model, logger) for p in classifier.pids)
    return_channel = remotecall_fetch_all(shards)
    train_loss, gradients = nothing, nothing
    for _ in 1:length(shards)
        p, r = take!(return_channel)
        if r !== nothing
            loss, grad = r
            if train_loss === nothing
                train_loss, gradients = loss, grad
            else
                # @show p
                # @show loss
                train_loss += loss
                for (p, dp) in zip(gradients.params, grad.params)
                    # @show size(gradients[p])
                    # @show size(grad[dp])
                    array = gradients[p]
                    array += grad[dp]
                end
            end
        end
    end
    @show train_loss
    return train_loss, reindex(gradients, weights)
end

function reindex(g, w)
    grads = IdDict()
    for (gp, wp) in zip(g.params, w)
        grads[wp] = g[gp]
    end
    return Zygote.Grads(grads, w)
end
