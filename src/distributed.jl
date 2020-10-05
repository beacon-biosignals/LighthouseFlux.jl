struct DistributedFluxClassifier <: AbstractFluxClassifier
    workerpool::AbstractWorkerPool
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
function loss_and_prediction_and_votes(model)
    for (dst, src) in zip(_model_params, Zygote.Params(Flux.params(model)))
        copyto!(dst, src)
    end
    batch, votes = try
        first(_test_batches)
    catch e
        @error "error taking _test_batches" exception=(e, catch_backtrace())
        return nothing
    end
    Flux.testmode!(_model)
    l, preds = loss_and_prediction(_model, batch...)
    return (l, preds, votes)
end

function loss_and_prediction_and_votes(classifier::DistributedFluxClassifier; timeout_secs=42.0)
    model_push!(classifier, :test; timeout_secs=timeout_secs)
    Channel(42) do channel
        try
            remote_channel = RemoteChannel(() -> channel)
            asyncmap(collect(classifier.workerpool)) do p
                Distributed.remotecall_eval(LighthouseFlux, p, quote
                    _predictions = $remote_channel
                    for (test_batch, votes_indices) in _test_batches
                        l, preds = loss_and_prediction(_model, test_batch...)
                        put!(_predictions, (l, preds, votes_indices))
                    end
                end)
            end
        catch e
            @error "error pushing predictions to _predictions" exception=(e, catch_backtrace())
        end
    end
end

function model_update!(model, trainmode)
    for (dst, src) in zip(_model_params, Zygote.Params(Flux.params(model)))
        copyto!(dst, src)
    end
    if mode == :train
        Flux.trainmode!(_model)
    elseif mode == :test
        Flux.testmode!(_model)
    else
        error("unknown model mode $mode not in [:train, :test]")
    end
    return nothing
end

function model_push!(model::DistributedFluxClassifier, mode; timeout_secs=42.0)
    mode ∉ [:train, test] && error("unknown model mode $mode not in [:train, :test]")
    shards = Dict{Int,Any}( p => (model_update!, Flux.cpu(classifier.model.model), mode) for p in classifier.workerpool.workers)
    return_channel = remotecall_fetch_all(shards)
    asyncmap(shards) do (pid, _)
        status = timedwait(() -> isready(return_channel), timeout_secs)
        if status != :ok
            @warn "timeout pushing model to pid $pid !!!"
        end
    end
end

function loss_and_gradient(model, logger::RemoteChannel)
    model_push!(model, :train)
    batch = try
        first(_training_batches)
    catch e
        @error "loss_and_gradient on worker" exception=(e, catch_backtrace())
        return nothing
    end
    train_loss, back = log_resource_info!(logger, "train/forward_pass";
                                          suffix="_per_batch") do
        f = () -> loss(_model, batch...)
        return Zygote.pullback(f, _model_params)
    end
    log_value!(logger, "train/loss_per_batch", train_loss)
    gradients = log_resource_info!(logger, "train/reverse_pass";
                                   suffix="_per_batch") do
        return back(Zygote.sensitivity(train_loss))
    end
    return (train_loss, [Flux.cpu(gradients[p]) for p in gradients.params])
end

function loss_and_gradient(classifier::DistributedFluxClassifier, weights, b, logger::RemoteChannel; timeout_secs=42.0)
    shards = Dict{Int,Any}( p => (loss_and_gradient, Flux.cpu(classifier.model.model), logger) for p in classifier.workerpool.workers)
    return_channel = remotecall_fetch_all(shards)
    train_loss, gradients, count = nothing, nothing, 0.0
    pids = []
    for (pid, _) in shards
        status = timedwait(() -> isready(return_channel), timeout_secs)
        if status == :ok
            p, r = take!(return_channel)
            # @show p, r
            push!(pids, pid)
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
        else
            pids = Set(pids)
            unresponsive = setdiff(classifier.workerpool.workers, pids)
            @warn "workers $unresponsive unresponsive, removing from worker pool, continuing tick without their batch data."
            classifier.workerpool.workers = pids
            for p in unresponsive
                @async try
                    rmprocs(p; waitfor=1)
                catch
                    rmprocs(p; waitfor=1)
                    nothing
                    # XXX call ClusterMangers kill
                end
            end
        end
    end
    # train_loss /= count
    # map(g -> g / count, gradients)
    @show train_loss
    return train_loss, reindex(gradients, weights)
end

function reindex(nope::Nothing, w)
    grads = IdDict()
    for wp in w
        grads[wp] = zero(wp)
    end
    return Zygote.Grads(grads, w)
end

function reindex(g::Vector, w)
    grads = IdDict()
    for (gp, wp) in zip(g, w)
        grads[wp] = gp
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
    # @info repr(losses)
    mean_loss = sum(losses) ./ length(losses)
    log_value!(logger, logger_prefix * "/mean_loss_per_epoch", mean_loss)
    return mean_loss
end

