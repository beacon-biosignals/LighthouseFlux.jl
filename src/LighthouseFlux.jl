module LighthouseFlux

using Zygote: Zygote
using Flux: Flux
using Lighthouse: Lighthouse, classes, log_resource_info!, log_values!, log_arrays!
using Functors
using Statistics

export FluxClassifier

#####
##### `FluxClassifier`
#####

struct FluxClassifier{M,O,C,P,OH,OC} <: Lighthouse.AbstractClassifier
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
    return FluxClassifier(Flux.testmode!(model), optimiser, classes, params, onehot,
                          onecold)
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

# Modified from `Functors.fmap`
"""
    fforeach_pairs(F, x, keys=(); exclude=Functors.isleaf, cache=IdDict(),
                   prune=Functors.NoKeyword(), combine=(ks, k) -> (ks..., k))

Walks the Functors.jl-compatible graph `x` (by calling `pairs ∘ Functors.children`), applying `F(parent_key, child)` at each step along the way. Here `parent_key` is the `key` part of a key-value pair returned from `pairs ∘ Functors.children`, combined with the previous `parent_key` by `combine`.

## Example

```jldoctest ex
julia> using Functors, LighthouseFlux

julia> struct Foo; x; y; end

julia> @functor Foo

julia> struct Bar; x; end

julia> @functor Bar

julia> m = Foo(Bar([1,2,3]), (4, 5, Bar(Foo(6, 7))));

julia> LighthouseFlux.fforeach_pairs((k,v) -> @show((k, v)), m)
(k, v) = ((:x,), Bar([1, 2, 3]))
(k, v) = ((:x, :x), [1, 2, 3])
(k, v) = ((:y,), (4, 5, Bar(Foo(6, 7))))
(k, v) = ((:y, 1), 4)
(k, v) = ((:y, 2), 5)
(k, v) = ((:y, 3), Bar(Foo(6, 7)))
(k, v) = ((:y, 3, :x), Foo(6, 7))
(k, v) = ((:y, 3, :x, :x), 6)
(k, v) = ((:y, 3, :x, :y), 7)
```

The `combine` argument can be used to customize how the keys are combined. For example

```jldoctest ex
julia> LighthouseFlux.fforeach_pairs((k,v) -> @show((k, v)), m, ""; combine=(ks, k) -> string(ks, "/", k))
(k, v) = ("/x", Bar([1, 2, 3]))
(k, v) = ("/x/x", [1, 2, 3])
(k, v) = ("/y", (4, 5, Bar(Foo(6, 7))))
(k, v) = ("/y/1", 4)
(k, v) = ("/y/2", 5)
(k, v) = ("/y/3", Bar(Foo(6, 7)))
(k, v) = ("/y/3/x", Foo(6, 7))
(k, v) = ("/y/3/x/x", 6)
(k, v) = ("/y/3/x/y", 7)

```

"""
function fforeach_pairs(F, x, keys=(); exclude=Functors.isleaf, cache=IdDict(),
                        prune=Functors.NoKeyword(), combine=(ks, k) -> (ks..., k))
    walk = (f, x) -> for (k, v) in pairs(Functors.children(x))
        F(combine(keys, k), v)
        f(k, v)
    end
    haskey(cache, x) && return prune isa Functors.NoKeyword ? cache[x] : prune
    cache[x] = exclude(x) ? (keys, x) :
               walk((k, x) -> fforeach_pairs(F, x, combine(keys, k); combine, exclude,
                                             cache, prune), x)
    return nothing
end

"""
    gather_weights_gradients(classifier, gradients)

Collects the weights and gradients from `classifier` into a `Dict`.
"""
function gather_weights_gradients(classifier, gradients)
    values = Dict{String, Any}()
    fforeach_pairs(classifier.model, "";
                combine=(ks, k) -> string(ks, "/", k)) do k, v
        if haskey(gradients, v)
            values[string("train/gradients", k)] = gradients[v]
        end
        if v isa AbstractArray
            values[string("train/weights", k)] = v
        end
    end
    return values
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

function Lighthouse.train!(classifier::FluxClassifier, batches, logger)
    Flux.trainmode!(classifier.model)
    weights = Zygote.Params(classifier.params)
    for (i, batch) in enumerate(batches)
        train_loss, back = log_resource_info!(logger, "train/forward_pass";
                                              suffix="_per_batch") do
            f = () -> loss(classifier.model, batch...)
            return Zygote.pullback(f, weights)
        end
        log_values!(logger, ("train/loss_per_batch" => train_loss,
                             "train/batch_index" => i))
        gradients = log_resource_info!(logger, "train/reverse_pass"; suffix="_per_batch") do
            return back(Zygote.sensitivity(train_loss))
        end
        log_resource_info!(logger, "train/update"; suffix="_per_batch") do
            Flux.Optimise.update!(classifier.optimiser, weights, gradients)
            return nothing
        end
        log_arrays!(logger, gather_weights_gradients(classifier, gradients))
    end
    Flux.testmode!(classifier.model)
    return nothing
end

function Lighthouse.loss_and_prediction(classifier::FluxClassifier, batch...)
    return Flux.cpu(loss_and_prediction(classifier.model, batch...))
end

#####
##### `CuIterator`
#####

Base.@deprecate_moved CuIterator CuArrays false

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
        @info output_size = size(input)
    end
    return input
end

end # module
