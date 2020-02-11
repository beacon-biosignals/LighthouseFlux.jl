module LighthouseFlux

using Lighthouse: log_resource_info!, log_value!
using Flux

# TODO
"""
- prediction via `(::C)(input_feature_batch::AbstractArray)::AbstractMatrix`

Calling `(model::C)(input_feature_batch::AbstractArray)` is expected to return
a matrix where each column is the `model`'s soft label prediction for the
corresponding sample in `input_feature_batch`.
"""
struct FluxClassifier <: Lighthouse.AbstractClassifier
    model::Any
    optimizer::Any
end

#
# TODO deprecate `default_optimizer`

function Lighthouse.train!(classifier::FluxClassifier, batches, logger)
    for batch in batches
        train_loss = log_resource_info!(logger, "training/forward_pass";
                                        suffix="_per_batch") do
            return loss(classifier.model, batch...)
        end
        log_value!(logger, "training/loss_per_batch", Flux.data(train_loss))
        gradients = log_resource_info!(logger, "training/reverse_pass";
                                       suffix="_per_batch") do
            return Flux.gradient(() -> train_loss, Flux.params(classifier.model))
        end
        log_resource_info!(logger, "training/update"; suffix="_per_batch") do
            Flux.Tracker.update!(classifier.optimizer, Flux.params(classifier.model), gradients)
            return nothing
        end
    end
    return nothing
end

#
# """
#     loss(model::AbstractClassifier,
#          input_feature_batch::AbstractArray,
#          args...)
#
# Return the loss of `model` applied to `input_feature_batch` given `args`.
#
# The last dimension of the given array arguments is the batch size, such that
# `size(input_feature_batch)[end] == size(soft_label_batch)[end]`.
#
# This method must be implemented for each `AbstractClassifier` subtype that wishes
# to support the `learn!` interface.
# """
# function loss end
#
# """
#     loss_and_prediction(model::AbstractClassifier, input_feature_batch, args...)
#
# Return `(loss(model, input_feature_batch, args...), model(input_feature_batch))`.
#
# Subtypes of `AbstractClassifier` may overload this function to remove redundant
# evaluations of `model` during execution of `learn!`.
# """
# function loss_and_prediction(model::AbstractClassifier, input_feature_batch, args...)
#     return loss(model, input_feature_batch, args...), model(input_feature_batch)
# end

# TODO Lighthouse.is_early_stopping_exception

#####
##### `Flux.Chain` debugging
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
