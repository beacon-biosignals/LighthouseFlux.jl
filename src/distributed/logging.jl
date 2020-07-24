
"""
    DistributedLogger(archive_path::String)

- receives and handles `Pair{String,Any}` logs through `inbox::RemoteChannel`
- archives logs into a sequence of `\$archive_path/1.bson`, `\$archive_path/2.bson` files
  of ~10Mb each
- forwards logs to `outbox::RemoteChannel` for immediate processing
"""
struct DistributedLogger
    archive_path::String
    inbox::RemoteChannel{Channel{Pair{String,Any}}}
    outbox::RemoteChannel{Channel{Pair{String,Any}}}
    logged::Dict{String, Vector{Any}}
    handler::Task
    function DistributedLogger(archive_path::String, inbox_size::Int=8192, outbox_size::Int=8192)
        mkpath(archive_path)
        inbox = RemoteChannel(() -> Channel{Pair{String,Any}}(inbox_size))
        outbox = RemoteChannel(() -> Channel{Pair{String,Any}}(outbox_size))
        logged = Dict{String, Vector{Any}}()
        unarchived = Dict{String, Vector{Any}}()
        handler = @async while true
            field, value = take!(inbox)
            push!(get!(() -> Any[], logged, field), value)
            push!(get!(() -> Any[], unarchived, field), value)
            put!(outbox, field => value)
            if rand(Float32) < 0.00042 && Base.summarysize(unarchived) > 1e7 # once in a while...
                n = length(readdir(archive_path)) + 1
                bson(joinpath(archive_path, "$n.bson"), unarchived)
                empty!(unarchived)
                rm(joinpath(archive_path, "tmp.bson"), force=true)
            end
        end
        return new(archive_path, inbox, outbox, logged, handler)
    end
end

function Base.close(logger::DistributedLogger)
    n = length(readdir(logger.archive_path)) + 1
    bson(joinpath(logger.archive_path, "$n.bson"), logger.logged)
    close(logger.inbox)
    close(logger.outbox)
end

# poached from ParallelDataTransfer.jl
function sendto(ps::AbstractVector{Int}; args...)
    for (nm, val) in args, p in ps
        @spawnat(p, Core.eval(Main, Expr(:(=), nm, val)))
    end
end

"""
    DistributedLogger(ps::AbstractVector{Int}, archive_path::String)

Construct a `DistributedLogger` and send its inbox to `ps` remote pids as symbol `Main.log_channel`,
so that remote workers can `put!(log_channel, field => value)`
"""
function DistributedLogger(ps::AbstractVector{Int}, archive_path::String, inbox_size=8192, outbox_size=8192)
    logger = DistributedLogger(archive_path, inbox_size)
    sendto(ps; log_channel=logger.inbox)
    return logger
end

inbox(logger::DistributedLogger) = logger.inbox
outbox(logger::DistributedLogger) = logger.outbox

#####
##### Interface similar to Lighthouse.LearnLogger
#####

function Lighthouse.log_plot!(logger::RemoteChannel, field::AbstractString, plot, plot_data)
    log_value!(logger, field, plot_data)
    log_value!(logger, "__plot__" * field, plot)
    return plot
end

function Lighthouse.log_event!(logger::RemoteChannel, value)
    logged = string(now(), " | ", value)
    put!(logger, "events" => logged)
    return logged
end

function Lighthouse.log_value!(logger::RemoteChannel, field::AbstractString, value)
    put!(logger, field => value)
    return value
end

# TODO replace this with `Lighthouse.log_resource_info!` generic in `logger`
function Lighthouse.log_resource_info!(logger::RemoteChannel, section::AbstractString,
                            info::Lighthouse.ResourceInfo; suffix::AbstractString="")
    log_value!(logger, section * "/time_in_seconds" * suffix, info.time_in_seconds)
    log_value!(logger, section * "/gc_time_in_seconds" * suffix, info.gc_time_in_seconds)
    log_value!(logger, section * "/allocations" * suffix, info.allocations)
    log_value!(logger, section * "/memory_in_mb" * suffix, info.memory_in_mb)
    return info
end

# TODO replace this with `Lighthouse.log_resource_info!` generic in `logger`
function Lighthouse.log_resource_info!(f, logger::RemoteChannel, section::AbstractString;
                            suffix::AbstractString="")
    result, resource_info = Lighthouse.call_with_resource_info(f)
    log_resource_info!(logger, section, resource_info; suffix=suffix)
    return result
end

"""
    read_log_archive(archive_path::String)

read archived logs into a single `Dict{String,Any}` containing vectors of values for each field.
"""
function read_log_archive(archive_path::String)
    logged = Dict{String,Any}()
    for file in readdir(archive_path, join=true, sort=true)
        d = BSON.load(file)
        for (field, values) in d
            previous = get!(() -> empty(values), logged, field)
            append!(previous, values)
        end
    end
    return logged
end

