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
function loss_and_prediction_and_votes(classifier::FluxClassifier)
    batch = try
        first(_test_batches)
    catch e
        @error "error taking _test_batches" exception=(e, catch_backtrace())
        return nothing
    end
    votes = try
        take!(_votes_indices_channel)
    catch e
        @error "error taking _votes_indices" exception=(e, catch_backtrace())
        return nothing
    end
    Flux.testmode!(classifier.model)
    l, preds = loss_and_prediction(classifier.model, batch...)
    return (l, preds, votes)
end

function loss_and_prediction_and_votes(classifier::DistributedFluxClassifier)
    shards = Dict{Int,Any}( p => (loss_and_prediction_and_votes, classifier.model) for p in classifier.pids)
    return_channel = remotecall_fetch_all(shards)
    results = Dict{Int,Any}( take!(return_channel) for _ in shards )
    return [results[p] for p in classifier.pids if results[p] !== nothing ]
end

function loss_and_gradient(classifier::FluxClassifier, logger::RemoteChannel)
    batch = try
        first(_training_batches)
    catch e
        @error "loss_and_gradient on worker" exception=(e, catch_backtrace())
        return nothing
    end
    train_loss, back = log_resource_info!(logger, "train/forward_pass";
                                          suffix="_per_batch") do
        f = () -> loss(classifier.model, batch...)
        weights = Zygote.Params(Flux.params(classifier.model))
        return Zygote.pullback(f, weights)
    end
    log_value!(logger, "train/loss_per_batch", train_loss)
    gradients = log_resource_info!(logger, "train/reverse_pass";
                                   suffix="_per_batch") do
        return back(Zygote.sensitivity(train_loss))
    end
    return (train_loss, [gradients[p] for p in gradients.params])
end

function loss_and_gradient(classifier::DistributedFluxClassifier, weights, b, logger::RemoteChannel)
    shards = Dict{Int,Any}( p => (loss_and_gradient, classifier.model, logger) for p in classifier.pids)
    return_channel = remotecall_fetch_all(shards)
    train_loss, gradients, count = nothing, nothing, 0.0
    for _ in 1:length(shards)
        p, r = take!(return_channel)
        if r !== nothing && eltype(r[2]) != Nothing
            loss, grad = r
            if train_loss === nothing
                train_loss, gradients = loss, grad
            else
                train_loss += loss
                count += 1.0
                map(+, gradients, grad)
            end
        end
    end
    # train_loss /= count
    # map(g -> g / count, gradients)
    @show train_loss
    return train_loss, reindex(gradients, weights)
end

function reindex(g::Vector, w)
    grads = IdDict()
    for (gp, wp) in zip(g, w)
        grads[wp] = gp
    end
    return Zygote.Grads(grads, w)
end

function reindex(g::Zygote.Grads, w)
    grads = IdDict()
    for (gp, wp) in zip(g.params, w)
        grads[wp] = g[gp]
    end
    return Zygote.Grads(grads, w)
end

function Lighthouse.predict!(model::DistributedFluxClassifier,
                             predicted_soft_labels::AbstractMatrix,
                             batches::UnitRange{Int}, logger;
                             logger_prefix::AbstractString)
    losses = []
    for b in batches
        for (batch_loss, soft_label_batch, votes) in loss_and_prediction_and_votes(model)
            for (i, soft_label) in enumerate(eachcol(soft_label_batch))
                predicted_soft_labels[votes[i], :] = soft_label
            end
            log_value!(logger, logger_prefix * "/loss_per_batch", batch_loss)
            push!(losses, batch_loss)
        end
    end
    @info repr(losses)
    mean_loss = sum(losses) ./ length(losses)
    log_value!(logger, logger_prefix * "/mean_loss_per_epoch", mean_loss)
    return mean_loss
end

