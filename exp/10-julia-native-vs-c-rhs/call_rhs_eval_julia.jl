using Libdl
using Printf
using Statistics
using Test

N_TRIALS = 3
N_RUNS = 41_000
N = 3200

include("rhsversions.jl")
using .RHSVersions
include("callback.jl")
using .CallbackWrapper

function benchmark_this_version(version_name, func, udot, u, p)
    runtimes = []
    dx = p[1]

    if version_name == "v5"
        # Warm up the function under benchmark.
        func(udot, u[:, 1], p, 0.0)
        for t = 1:N_TRIALS
            tic = time_ns()
            for j = 1:N_RUNS
                func(udot, u[:, 1], (dx,), 0.0)
            end
            toc = time_ns()
            elapsed = (toc - tic) / 1e9
            push!(runtimes, elapsed)
        end
    elseif version_name == "cwrapper"
        # Warm up the function under benchmark.
        func(0.0, u[:, 1], udot, p)
        for t = 1:N_TRIALS
            tic = time_ns()
            for j = 1:N_RUNS
                func(0.0, u[:, 1], udot, (dx,))
            end
            toc = time_ns()
            elapsed = (toc - tic) / 1e9
            push!(runtimes, elapsed)
        end
    else
        println("BAD!")
    end

    mean, ci = runtime_stats(runtimes)
    label = @sprintf "Julia, %s" version_name
    print_runtime(label, mean, ci)
end

function runtime_stats(elapsed_times)
    runtime_mean = mean(elapsed_times)
    runtime_std = std(elapsed_times; corrected=true, mean=runtime_mean)
    sem = runtime_std / sqrt(length(elapsed_times))
    ci = 2 * sem

    return runtime_mean, ci
end

function print_runtime(prefix, mean, ci)
    @printf "%-32s %.3f ± %.3f\n" prefix mean ci
end

function measure()
    x = collect(range(0, 2, N + 1))
    dx = 2 / N
    u0 = 0.5 .- 0.25 * sin.(pi * x)

    CFL = 0.5
    dt_max = dx * CFL

    t0 = 0.0
    tfinal = 10.0
    p = (dx, )

    libhandle = Libdl.dlopen("burgers.so")
    compute_rhs_fn = Libdl.dlsym(libhandle, "rhs")
    println(compute_rhs_fn)
    compute_rhs_wrapper = CallbackWrapper.make_wrapper_over_c_callback(compute_rhs_fn)

    udot = similar(u0)
    udot_v1 = similar(u0)
    udot_v2 = similar(u0)
    udot_v3 = similar(u0)
    udot_v4 = similar(u0)
    udot_v5 = similar(u0)
    udot_cwrapper = similar(u0)
    u = rand(N + 1, N_RUNS)
    # We use deterministic input because in the end we print the leftmost
    # value to compare with results from Python.
    for j = 1:N_RUNS
        u[:, j] = u0
    end

    @printf "Julia, accumulated runtime of %d RHS evals, statistics from %d trials\n" N_RUNS N_TRIALS
    @printf "Problem size is %d\n" length(udot)

    benchmark_this_version("v5", compute_rhs_v5, udot_v5, u, p)
    benchmark_this_version("cwrapper", compute_rhs_wrapper, udot_cwrapper, u, p)

    @test udot_v5 ≈ udot_cwrapper rtol=1e-14 atol=1e-14
    @printf "Leftmost udot_1 value: %.16f\n" udot_v5[1]
    @printf "Leftmost udot_2 value: %.16f\n" udot_cwrapper[1]
end

measure()
