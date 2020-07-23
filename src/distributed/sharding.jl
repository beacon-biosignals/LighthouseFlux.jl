
function remotecall_fetch_all(objects::Dict{Int,T}) where T
    channel = Channel(length(objects))
    for (p, obj) in objects
        @async try
            result = remotecall_fetch(first(obj), p, Iterators.drop(obj, 1)...)
            put!(channel, p => result)
        catch e
            @error "Remote error at pid $p" exception=(e, catch_backtrace())
            println(e)
        end
    end
    return channel
    # return Iterators.take(channel, length(objects))
end

# poached from ParallelDataTransfer.jl
macro defineat(pids,ex,mod=Main)
    quote
        for p in $(esc(pids))
            remotecall_wait(p,$(esc(mod)),$(QuoteNode(ex))) do mod,ex
                Core.eval(mod,ex)
            end
        end
    end
end
