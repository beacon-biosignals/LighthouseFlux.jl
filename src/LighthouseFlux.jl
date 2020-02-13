module LighthouseFlux

using Zygote: Zygote
using Flux: Flux
using CuArrays: CuArray, unsafe_free!, adapt
using Lighthouse: Lighthouse
using Lighthouse: classes, log_resource_info!, log_value!

#####
##### `FluxClassifier`
#####

# TODO docs
"""
- `Flux.params`
- `loss`
- `loss_and_prediction`
"""
struct FluxClassifier{M,O,C} <: Lighthouse.AbstractClassifier
    model::M
    optimizer::O
    classes::C
end

# TODO redo docs
"""
    loss(model, input_feature_batch::AbstractArray, args...)

Return the loss of `model` applied to `input_feature_batch` given `args`.

The last dimension of the given array arguments is the batch size, such that
`size(input_feature_batch)[end] == size(soft_label_batch)[end]`.

This method must be implemented for each `AbstractClassifier` subtype that
wishes to support the `learn!` interface.
"""
function loss end

# TODO redo docs
"""
    loss_and_prediction(model, input_feature_batch, args...)
"""
function loss_and_prediction end

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
    weights = Zygote.Params(Flux.params(classifier.model))
    for batch in batches
        train_loss, back = log_resource_info!(logger, "training/forward_pass";
                                              suffix="_per_batch") do
            f = () -> loss(classifier.model, batch...)
            return Zygote.pullback(f, weights)
        end
        log_value!(logger, "training/loss_per_batch", train_loss)
        gradients = log_resource_info!(logger, "training/reverse_pass";
                                       suffix="_per_batch") do
            return back(Zygote.sensitivity(train_loss))
        end
        log_resource_info!(logger, "training/update"; suffix="_per_batch") do
            Flux.update!(classifier.optimizer, weights, gradients)
            return nothing
        end
    end
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
