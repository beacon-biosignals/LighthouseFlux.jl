using Test, StableRNGs
using LighthouseFlux, Lighthouse, Flux

# Set up plotting backend for Plots.jl (GR)
using Lighthouse.Plots
gr()
GR.inline("png")

mutable struct TestModel
    chain::Chain
end

(m::TestModel)(X) = m.chain(X)

Flux.@functor TestModel (chain,)

function LighthouseFlux.loss(m::TestModel, input_feature_batch, soft_label_batch)
    return Flux.mse(m(input_feature_batch), soft_label_batch)
end

@testset "learn!(::TestModel, ...)" begin
    mktempdir() do tmpdir
        rng = StableRNG(157)
        classes = ["class_$i" for i in 1:5]
        c, n = 5, 3
        model = TestModel(Chain(Dense(4 * c, 2 * c),
                                Dense(2 * c, c), softmax))
        # assure that `testmode!` and `trainmode!` is being utilized correctly after training
        test_input = rand(StableRNG(42), Float32, 4 * c) # get test input
        y_pretrained = model(test_input) # test model output before being trained
        classifier = FluxClassifier(model, ADAM(0.1), classes)
        train_batches = [(rand(rng, 4 * c, n), rand(rng, 1, n)) for _ in 1:100]
        test_batches = [((rand(rng, 4 * c, n), rand(rng, 1, n)), (n * i - n + 1):(n * i))
                        for i in 1:10]
        possible_vote_labels = collect(0:length(classes))
        votes = [rand(rng, possible_vote_labels) for sample in 1:(n * 10), voter in 1:7]
        logger = Lighthouse.LearnLogger(joinpath(tmpdir, "logs"), "test_run")
        limit = 4
        let counted = 0
            upon_loss_decrease = Lighthouse.upon(logger,
                                                 "test_set_prediction/mean_loss_per_epoch";
                                                 condition=<, initial=Inf)
            callback = n -> begin
                upon_loss_decrease() do _
                    counted += n
                end
            end
            elected = majority.((rng,), eachrow(votes), (1:length(classes),))
            Lighthouse.learn!(classifier, logger, () -> train_batches, () -> test_batches,
                              votes, elected; epoch_limit=limit, post_epoch_callback=callback)
            # NOTE: the RNG chosen above just happens to allow this to work every time,
            # since the loss happens to actually "improve" on the random data each epoch
            @test counted == sum(1:limit)
        end
        for key in [
            "train/loss_per_batch"
            "train/forward_pass/time_in_seconds_per_batch"
            "train/forward_pass/gc_time_in_seconds_per_batch"
            "train/forward_pass/allocations_per_batch"
            "train/forward_pass/memory_in_mb_per_batch"
            "train/reverse_pass/time_in_seconds_per_batch"
            "train/reverse_pass/gc_time_in_seconds_per_batch"
            "train/reverse_pass/allocations_per_batch"
            "train/reverse_pass/memory_in_mb_per_batch"
            "train/update/time_in_seconds_per_batch"
            "train/update/gc_time_in_seconds_per_batch"
            "train/update/allocations_per_batch"
            "train/update/memory_in_mb_per_batch"
        ]
            @test length(logger.logged[key]) == length(train_batches) * limit
        end
        for key in [
            "test_set_prediction/loss_per_batch"
            "test_set_prediction/time_in_seconds_per_batch"
            "test_set_prediction/gc_time_in_seconds_per_batch"
            "test_set_prediction/allocations_per_batch"
            "test_set_prediction/memory_in_mb_per_batch"
        ]
            @test length(logger.logged[key]) == length(test_batches) * limit
        end
        for key in [
            "test_set_prediction/mean_loss_per_epoch"
            "test_set_evaluation/time_in_seconds_per_epoch"
            "test_set_evaluation/gc_time_in_seconds_per_epoch"
            "test_set_evaluation/allocations_per_epoch"
            "test_set_evaluation/memory_in_mb_per_epoch"
        ]
            @test length(logger.logged[key]) == limit
        end
        @test length(logger.logged["test_set_evaluation/metrics_per_epoch"]) == limit

        # test `testmode!` is correctly utiltized
        y_post_train = model(test_input)
        @test y_post_train != y_pretrained # check if model has trained
        @test y_post_train == model(test_input) # make sure model output is determinsitic

        # test onehot/onecold overloads
        classifier = FluxClassifier(model, ADAM(0.1), classes;
                                    onehot=(x -> fill(x, length(classes))), onecold=sum)
        @test Lighthouse.onehot(classifier, 3) == fill(3, length(classes))
        @test Lighthouse.onecold(classifier, [0.31, 0.43, 0.13]) == 0.87
    end
end

@testset "`evaluate_chain_in_debug_mode`" begin
    chain = Chain(Dense(4, 2), Dense(2, 1))
    input = Float32[
        1 2 3 4
        5 6 7 8
        8 7 6 5
        4 3 2 1
    ]
    @test @test_logs((:info, "Executing layer 1 / 2..."), (:info, (2, 4)),
                     (:info, "Executing layer 2 / 2..."), (:info, (1, 4)),
                     LighthouseFlux.evaluate_chain_in_debug_mode(chain, input)) ≈
          chain(input)
end
