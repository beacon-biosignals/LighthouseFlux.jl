
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

remotecall_channel(f, l, wp::AbstractWorkerPool, mod::Module=Main, size=2) = remotecall_channel(f, l, collect(wp.workers), mod; size=size)

function remotecall_channel(f, l, pids::AbstractVector{Int}=[1]; mod::Module=Main, size=2)
    inbox = RemoteChannel(() -> Channel(size) do c
        foreach(x -> put!(c, x), l)
    end)
    Channel(size) do channel
        try
            outbox = RemoteChannel(() -> channel)
            asyncmap(pids) do p
                Distributed.remotecall_eval(mod, p, quote
                    while true
                        try
                            x = take!($inbox)
                            put!($outbox, $f(x...))
                        catch e
                            (isa(e, InvalidStateException) || (isa(e, Distributed.RemoteException) && isa(e.captured.ex, InvalidStateException))) && return
                            throw(e)
                        end
                    end
                end)
            end
        catch e
            @error "error in remotecall_channel" exception=(e, catch_backtrace())
        end
    end
end

