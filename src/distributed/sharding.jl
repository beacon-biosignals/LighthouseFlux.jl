
function shard(ps::AbstractVector{Int}, xs...)
    xs = collect(xs)
    # sort both for shard consistency
    sort!(xs)
    sort!(ps)
    return Dict{Int,Any}(p => part for (part, p) in zip(Iterators.partition(xs, length(xs) ÷ length(ps)), ps))
end

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

