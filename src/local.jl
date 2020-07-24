
#####
##### `FluxClassifier`
#####

struct FluxClassifier{M,O,C,P,OH,OC} <: AbstractFluxClassifier
    model::M
    optimiser::O
    classes::C
    params::P
    onehot::OH
    onecold::OC
end

"""
    FluxClassifier(model, optimiser, classes; params=Flux.params(model),
                   onehot=(label -> Flux.onehot(label, 1:length(classes))),
                   onecold=(label -> Flux.onecold(label, 1:length(classes))))

Return a `FluxClassifier <: Lighthouse.AbstractClassifier` with the given arguments:

- `model`: a Flux model. The model must additionally support LighthouseFlux's [`loss`](@ref)
  and [`loss_and_prediction`](@ref) functions.

- `optimiser`: a [Flux optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/)

- `classes`: a `Vector` or `Tuple` of possible class values; this is the return
value of `Lighthouse.classes(::FluxClassifier)`.

- `params`: The parameters to optimise during training; generally, a `Zygote.Params`
value or a value that can be passed to `Zygote.Params`.

- `onehot`: the function used to convert hard labels to soft labels when
`Lighthouse.onehot` is called with this classifier.

- `onecold`: the function used to convert soft labels to hard labels when
`Lighthouse.onecold` is called with this classifier.
"""
function FluxClassifier(model, optimiser, classes; params=Flux.params(model),
                        onehot=(label -> Flux.onehot(label, 1:length(classes))),
                        onecold=(label -> Flux.onecold(label, 1:length(classes))))
    return FluxClassifier(Flux.testmode!(model), optimiser, classes, params, onehot, onecold)
end

"""
    loss(model, batch_arguments...)

Return the scalar loss of `model` given `batch_arguments`.

This method must be implemented for all `model`s passed to [`FluxClassifier`](@ref).
"""
function loss end

"""
    assemble_batch(batch)

Prepares a batch. Fallback impl is the identity.
"""
assemble_batch(batch) = batch

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

function Lighthouse.loss_and_prediction(classifier::FluxClassifier, batch...)
    return Flux.cpu(loss_and_prediction(classifier.model, batch...))
end

#####
##### Lighthouse `AbstractClassifier` Interface
#####

Lighthouse.classes(classifier::FluxClassifier) = classifier.classes


function Lighthouse.is_early_stopping_exception(::FluxClassifier, exception)
    return exception isa Flux.Optimise.StopException
end

Lighthouse.onehot(classifier::FluxClassifier, label) = classifier.onehot(label)

Lighthouse.onecold(classifier::FluxClassifier, label) = classifier.onecold(label)

#####
##### LighthouseFLux `AbstractFluxClassifier` Interface
#####

model(classifier::FluxClassifier) = classifier.model
params(classifier::FluxClassifier) = classifier.params
optimizer(classifier::FluxClassifier) = classifier.optimiser

function loss_and_gradient(classifier::FluxClassifier, weights, batchspec, logger::Lighthouse.LearnLogger)
    batch = assemble_batch(batchspec)
    train_loss, back = log_resource_info!(logger, "train/forward_pass";
                                          suffix="_per_batch") do
        f = () -> loss(LighthouseFlux.model(classifier), batch...)
        return Zygote.pullback(f, weights)
    end
    log_value!(logger, "train/loss_per_batch", train_loss)
    gradients = log_resource_info!(logger, "train/reverse_pass";
                                   suffix="_per_batch") do
        return back(Zygote.sensitivity(train_loss))
    end
    return train_loss, gradients
end

#####
##### 
#####

function Lighthouse.train!(classifier::AbstractFluxClassifier, batches, logger)
    Flux.trainmode!(LighthouseFlux.model(classifier))
    weights = Zygote.Params(LighthouseFlux.params(classifier))
    @info "Starting train! loop"
    for batch in batches
        train_loss, gradients = loss_and_gradient(classifier, weights, batch, logger)
        log_resource_info!(logger, "train/update"; suffix="_per_batch") do
            Flux.Optimise.update!(optimizer(classifier), weights, gradients)
            return nothing
        end
    end
    Flux.testmode!(LighthouseFlux.model(classifier))
    return nothing
end

function Lighthouse.loss_and_prediction(classifier::AbstractFluxClassifier, batch...)
    return Flux.cpu(loss_and_prediction(LighthouseFlux.model(classifier), batch...))
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

