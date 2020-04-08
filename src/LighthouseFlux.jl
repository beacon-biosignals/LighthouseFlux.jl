module LighthouseFlux

using Zygote: Zygote
using Flux: Flux
using CuArrays: CuArray, unsafe_free!, adapt
using Lighthouse: Lighthouse, classes, log_resource_info!, log_value!

export FluxClassifier

#####
##### `FluxClassifier`
#####

struct FluxClassifier{M,O,C,P} <: Lighthouse.AbstractClassifier
    model::M
    optimiser::O
    classes::C
    params::P
end

"""
    FluxClassifier(model, optimiser, classes; params=Flux.params(model))

Return a `FluxClassifier <: Lighthouse.AbstractClassifier` with the given arguments:

- `model`: a Flux model. The model must additionally support LighthouseFlux's [`loss`](@ref)
  and [`loss_and_prediction`](@ref) functions.

- `optimiser`: a [Flux optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/)

- `classes`: a `Vector` or `Tuple` of possible class values; this is the return
value of `Lighthouse.classes(::FluxClassifier)`.

- `params`: The parameters to optimise during training; generally, a `Zygote.Params`
value or a value that can be passed to `Zygote.Params`.
"""
function FluxClassifier(model, optimiser, classes; params=Flux.params(model))
    return FluxClassifier(Flux.testmode!(model), optimiser, classes, params)
end

"""
    loss(model, batch_arguments...)

Return the scalar loss of `model` given `batch_arguments`.

This method must be implemented for all `model`s passed to [`FluxClassifier`](@ref).
"""
function loss end

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
function loss_and_prediction(model, input_batch, other_batch_arguments...)
    return (loss(model, input_batch, other_batch_arguments...), model(input_batch))
end

#####
##### Lighthouse `AbstractClassifier` Interface
#####

Lighthouse.classes(classifier::FluxClassifier) = classifier.classes

function Lighthouse.is_early_stopping_exception(::FluxClassifier, exception)
    return exception isa Flux.Optimise.StopException
end

function Lighthouse.onehot(classifier::FluxClassifier, hard_label)
    return Flux.onehot(hard_label, 1:length(classes(classifier)))
end

function Lighthouse.train!(classifier::FluxClassifier, batches, logger)
    Flux.trainmode!(classifier.model)
    weights = Zygote.Params(classifier.params)
    predictions = # TODO define struture
    votes = # TODO define votes - where this thing comes from might require external changes...
    for batch in batches
        train_loss, back = log_resource_info!(logger, "train/forward_pass";
                                              suffix="_per_batch") do
            if classifier.train_set_evaluation
                # TODO save `batch`'s labels to `votes` - how this happens
                # might require new interface constraints on `batch`...
                f = () -> begin
                    ls, pred = loss_and_prediction(classifier.model, batch...)
                    # TODO save `pred` to `predictions` + `votes`
                    return ls
                end
            else
                f = () -> loss(classifier.model, batch...)
            end
            return Zygote.pullback(f, weights)
        end
        log_value!(logger, "train/loss_per_batch", train_loss)
        gradients = log_resource_info!(logger, "train/reverse_pass";
                                       suffix="_per_batch") do
            return back(Zygote.sensitivity(train_loss))
        end
        log_resource_info!(logger, "train/update"; suffix="_per_batch") do
            Flux.Optimise.update!(classifier.optimiser, weights, gradients)
            return nothing
        end
    end
    if classifier.train_set_evaluation
        # TODO call Lighthouse.test_set_evaluation(...)
    end
    Flux.testmode!(classifier.model)
    return nothing
end

function Lighthouse.loss_and_prediction(classifier::FluxClassifier, batch...)
    return loss_and_prediction(classifier.model, batch...)
end

#####
##### `CuIterator`
#####
# ripped from https://github.com/JuliaGPU/CuArrays.jl/pull/467; this will live
# here so we can use it while the official version is vetted by the community
# NOTE: This code isn't testable on Travis CI, so it hurts coverage metrics.
# This will no longer be a problem once this code properly lives in CuArrays.

mutable struct CuIterator{B}
    batches::B
    previous::Any
    CuIterator(batches) = new{typeof(batches)}(batches)
end

function Base.iterate(c::CuIterator, state...)
    item = iterate(c.batches, state...)
    isdefined(c, :previous) && foreach(unsafe_free!, c.previous)
    item === nothing && return nothing
    batch, next_state = item
    cubatch = map(x -> adapt(CuArray, x), batch)
    c.previous = cubatch
    return cubatch, next_state
end

#####
##### miscellaneous utilities
#####

"""
    evaluate_chain_in_debug_mode(chain::Flux.Chain, input)

Evaluate `chain(input)`, printing additional debug information at each layer.
"""
function evaluate_chain_in_debug_mode(chain::Flux.Chain, input)
    for (i, layer) in enumerate(chain)
        @info "Executing layer $i / $(length(chain))..." layer size(input)
        input = layer(input)
        @info output_size=size(input)
    end
    return input
end

end # module
