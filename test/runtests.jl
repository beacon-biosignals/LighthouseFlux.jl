using Test, Random
using LighthouseFlux, Lighthouse, Flux

mutable struct TestModel
    chain::Chain
end

(m::TestModel)(X) = m.chain(X)

Flux.@treelike TestModel (chain,)

function LighthouseFlux.loss(m::TestModel, input_feature_batch, soft_label_batch)
    return Flux.mse(m(input_feature_batch), soft_label_batch)
end

@testset "learn!(::TestModel, ...)" begin
    mktempdir() do tmpdir
        rng = MersenneTwister(43)
        classes = ["class_$i" for i in 1:5]
        c, n = 5, 3
        model = TestModel(Chain(Dense(4*c, 2*c, initW=ones, initb=zeros),
                                Dense(2*c, c, initW=ones, initb=zeros),
                                softmax))
        classifier = FluxClassifier(model, ADAM(0.1), classes)
        training_batches = [(rand(rng, 4*c, n), rand(rng, 1, n)) for _ in 1:100]
        validation_batches = [((rand(rng, 4*c, n), rand(rng, 1, n)), (n*i - n + 1):(n*i)) for i in 1:10]
        possible_vote_labels = collect(0:length(classes))
        votes = [rand(rng, possible_vote_labels) for sample in 1:(n*10), voter in 1:7]
        logger = Lighthouse.LearnLogger(joinpath(tmpdir, "logs"), "test_run")
        limit = 5
        let counted = 0
            trigger = Lighthouse.on_change("validation/mean_loss_per_epoch", (_, n) -> (counted+=n; @info counted n))
            callback = (_, epoch, logger) -> trigger(epoch, logger)
            Lighthouse.learn!(classifier, logger, () -> training_batches, () -> validation_batches,
                              votes; epoch_limit=limit, post_epoch_callback=callback)
            # NOTE: the RNG chosen above just happens to allow this to work every time,
            # since the loss happens to actually "improve" on the random data each epoch
            @test counted == sum(1:limit)
        end
        for key in ["training/loss_per_batch"
                    "training/forward_pass/time_in_seconds_per_batch"
                    "training/forward_pass/gc_time_in_seconds_per_batch"
                    "training/forward_pass/allocations_per_batch"
                    "training/forward_pass/memory_in_mb_per_batch"
                    "training/reverse_pass/time_in_seconds_per_batch"
                    "training/reverse_pass/gc_time_in_seconds_per_batch"
                    "training/reverse_pass/allocations_per_batch"
                    "training/reverse_pass/memory_in_mb_per_batch"
                    "training/update/time_in_seconds_per_batch"
                    "training/update/gc_time_in_seconds_per_batch"
                    "training/update/allocations_per_batch"
                    "training/update/memory_in_mb_per_batch"]
            @test length(logger.logged[key]) == length(training_batches)*limit
        end
        for key in ["validation/loss_per_batch"
                    "validation/time_in_seconds_per_batch"
                    "validation/gc_time_in_seconds_per_batch"
                    "validation/allocations_per_batch"
                    "validation/memory_in_mb_per_batch"]
            @test length(logger.logged[key]) == length(validation_batches)*limit
        end
        for key in ["validation/mean_loss_per_epoch"
                    "evaluation/time_in_seconds_per_epoch"
                    "evaluation/gc_time_in_seconds_per_epoch"
                    "evaluation/allocations_per_epoch"
                    "evaluation/memory_in_mb_per_epoch"]
            @test length(logger.logged[key]) == limit
        end
        @test length(logger.logged["evaluation/metrics_per_epoch"]) == limit
    end
end

@testset "`evaluate_chain_in_debug_mode`" begin
    chain = Chain(Dense(4, 2), Dense(2, 1))
    input = Float32[1 2 3 4
                    5 6 7 8
                    8 7 6 5
                    4 3 2 1]
    @test @test_logs((:info, "Executing layer 1 / 2..."),
                     (:info, (2, 4)),
                     (:info, "Executing layer 2 / 2..."),
                     (:info, (1, 4)),
                     LighthouseFlux.evaluate_chain_in_debug_mode(chain, input)) ≈ chain(input)
end
