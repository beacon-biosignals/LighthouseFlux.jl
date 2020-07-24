
# a stripped down version of this should live in FastS3: a generic buffered reader
function buffered_batch_loader(batchspecs, batch_assembler; buffer_size=2)
    future_buffer = Channel(buffer_size) # using this channel as a blocking glorified FIFO / circular buffer..
    for i in 1:buffer_size
        put!(future_buffer, @async batch_assembler(batchspecs[i]...))
    end
    return Channel(buffer_size) do channel
        try
            for batchspec in Iterators.drop(batchspecs, buffer_size)
                future = take!(future_buffer)
                put!(future_buffer, batch_assembler(batchspec...))
                put!(channel, fetch(future))
            end
            for i in 1:buffer_size
                future = take!(future_buffer)
                put!(channel, fetch(future))
            end
        catch e
            @error "Error assembling batch" exception=(e, catch_backtrace())
            println(e)
        end
    end
end
