# from https://github.com/FluxML/model-zoo/blob/b4732e5a3158391f2fd737470ff63986420e42cd/vision/mnist/conv.jl

# Classifies MNIST digits with a convolutional network.
# Writes out saved model to the file "mnist_conv.bson".
# Demonstrates basic model construction, training, saving,
# conditional early-exit, and learning rate scheduling.
#
# This model, while simple, should hit around 99% test
# accuracy after training for approximately 20 epochs.

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy
using Base.Iterators: partition
using Printf
using CUDA
using LighthouseFlux, Lighthouse
using TensorBoardLogger
using Dates

# for headless plotting with GR
ENV["GKSwstype"]="100"

if has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

Base.@kwdef mutable struct Args
    lr::Float64 = 3e-3
    epochs::Int = 20
    batch_size = 128
    savepath::String = joinpath(@__DIR__, "..", "logs", "run")
    run_name::String = "abc"
    logger = LearnLogger(savepath, run_name)
end

# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

function get_processed_data(args)
    # Load labels and images from Flux.Data.MNIST
    train_labels = MNIST.labels()
    train_imgs = MNIST.images()
    mb_idxs = partition(1:length(train_imgs), args.batch_size)
    train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs] 
    
    # Prepare test set as one giant minibatch:
    test_imgs = MNIST.images(:test)
    test_labels = MNIST.labels(:test)
    test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))

    return train_set, test_set, test_labels
end

function make_rater_labels(true_labels; error_rate = 0.1, n_classes = 10)
    out_labels = similar(true_labels)
    for i = eachindex(out_labels, true_labels)
        if rand() < error_rate
            out_labels[i] = mod(true_labels[i] + 1, n_classes)
        else
            out_labels[i] = true_labels[i]
        end
    end
    return out_labels
end

# Build model

struct SimpleModel{C}
    chain::C
end

function SimpleModel(; imgsize = (28,28,1), nclasses = 10)
    cnn_output_size = Int.(floor.([imgsize[1]/8,imgsize[2]/8,32]))	

    chain = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), imgsize[3]=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    flatten,
    Dense(prod(cnn_output_size), 10))
    chain = gpu(chain)
    return SimpleModel{typeof(chain)}(chain)
end

Flux.@functor SimpleModel (chain,)

# make callable
(sm::SimpleModel)(args...) = sm.chain(args...)

# We augment `x` a little bit here, adding in random noise. 
augment(x) = x .+ gpu(0.1f0*randn(eltype(x), size(x)))

# Returns a vector of all parameters used in model
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

# Function to check if any element is NaN or not
anynan(x) = any(isnan.(x))

accuracy(x, y, model) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))


function LighthouseFlux.loss_and_prediction(model::SimpleModel, x, y)
    # We augment the data
    # a bit, adding gaussian random noise to our image to make it more robust.
    x̂ = augment(x)

    ŷ = model(x̂) # prediction

    # actually, ignore the model, and output y + 1 with 10% probability
    # mask = rand(length(y)) .< 0.1
    # ŷ = y + mask

    return logitcrossentropy(ŷ, y), ŷ
end

LighthouseFlux.loss(model::SimpleModel, x, y) = LighthouseFlux.loss_and_prediction(model, x, y)[1]

function train(; kws...)	
    args = Args(; kws...)

    _info_and_log = (msg::String) -> begin
        msg = Dates.format(now(), "HH:MM:SS ") * msg
        @info msg
        Lighthouse.log_event!(args.logger, msg)
        return nothing
    end


    isdir(args.savepath) || mkpath(args.savepath)

    _info_and_log("Loading data set")
    train_set, test_set, test_labels = get_processed_data(args)

    # Define our model.  We will use a simple convolutional architecture with
    # three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense layer.
    _info_and_log("Building model...")
    model = SimpleModel() 

    # Load model and datasets onto GPU, if enabled
    train_set = gpu.(train_set)
    test_set = gpu.(test_set)
    
    # Make sure our model is nicely precompiled before starting our training loop
    model(train_set[1][1])
	
    # Train our model with the given training set using the ADAM optimizer and
    # printing out performance against the test set as we go.
    opt = ADAM(args.lr)
    
    classifier = FluxClassifier(model, opt,  ["class_$i" for i in 0:9])
    _info_and_log("Beginning `learn!`...")

    votes = reduce(hcat, [ make_rater_labels(test_labels, error_rate = 0.1) for _ = 1:5 ])

    learn!(classifier, args.logger,
       () -> train_set,  () -> [(test_set, 1:length(test_labels))], votes)

    return cpu.(params(model))

    # the following is dead code, from the original model zoo example
    # I haven't deleted it yet because I wanted to port the functionality to 
    # Lighthouse callbacks, to show how the same loop can be done with Lighthouse

    _info_and_log("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0
    best_params = cpu.(params(model))

    for epoch_idx in 1:args.epochs
        # Train for a single epoch
        Lighthouse.train!(classifier, train_set, args.logger)
	    
        # Terminate on NaN
        if anynan(paramvec(model))
            @error "NaN params"
            break
        end
	
        # Calculate accuracy:
        acc = accuracy(test_set..., model)
		
        _info_and_log(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            _info_and_log(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end
	
        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            _info_and_log("Best epoch yet (epoch $(epoch_idx))")
            best_params = cpu.(params(model))
            best_acc = acc
            last_improvement = epoch_idx
        end
	
        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10.0
            _info_and_log(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
   
            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end
	
        if epoch_idx - last_improvement >= 10
            _info_and_log(" -> We're calling this converged.")
            break
        end
    end
    return best_params
end

# Testing the model, from saved model
function test(params; kws...)
    args = Args(; kws...)
    
    # Loading the test data
    _,test_set = get_processed_data(args)
    
    # Re-constructing the model with random initial weights
    model = SimpleModel()
        
    # Loading parameters onto the model
    Flux.loadparams!(model, params)
    
    test_set = gpu.(test_set)
    model = gpu(model)
    @show accuracy(test_set...,model)
end

best_params = train(; epochs=1)
test(best_params)
