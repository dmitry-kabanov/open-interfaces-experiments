using Libdl
using OrdinaryDiffEq
using Printf
using Statistics
using Test

N_TRIALS = 2
N_RUNS = 41_000
N = 3200

include("rhsversions.jl")
using .RHSVersions
include("callback.jl")
using .CallbackWrapper

function wrap_for_odejl(func)
    function wrapper(udot, u, p, t)
        func(t, u, udot, p)
    end

    return wrapper
end

function benchmark_this_version(version_name, func, u0, tspan, p)
    runtimes = []
    times = collect(range(tspan[1], tspan[2], 101))

    for t = 1:N_TRIALS
        odeProblem = ODEProblem(func, u0, tspan, p)
        solver = init(odeProblem, DP5(); reltol=1e-6, abstol=1e-12, save_everystep = false)
        # Warm up the function under benchmark.
        # func(udot, u0, p, 0.0)
        tic = time_ns()
        for t in times[2:end]
            step!(solver, t - solver.t, true)
        end
        toc = time_ns()
        elapsed = (toc - tic) / 1e9
        push!(runtimes, elapsed)
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
    tspan = (t0, tfinal)

    libhandle = Libdl.dlopen("burgers.so")
    compute_rhs_oif_fn = Libdl.dlsym(libhandle, "rhs_oif")
    compute_rhs_carray_fn = Libdl.dlsym(libhandle, "rhs_carray")
    compute_rhs_oif_wrapper = CallbackWrapper.make_wrapper_over_oif_c_callback(compute_rhs_oif_fn)
    compute_rhs_carray_wrapper = CallbackWrapper.make_wrapper_over_carray_c_callback(compute_rhs_carray_fn)

    udot = similar(u0)
    udot_v1 = similar(u0)
    udot_v2 = similar(u0)
    udot_v3 = similar(u0)
    udot_v4 = similar(u0)
    udot_v5 = similar(u0)
    udot_oif_cwrapper = similar(u0)
    udot_carray_cwrapper = similar(u0)
    u = rand(N + 1, N_RUNS)
    # We use deterministic input because in the end we print the leftmost
    # value to compare with results from Python.
    for j = 1:N_RUNS
        u[:, j] = u0
    end

    @printf "Solving ODEs, statistics from %d trials\n" N_TRIALS
    @printf "Problem size is %d\n" length(udot)

    w1 = wrap_for_odejl(compute_rhs_oif_wrapper)
    w2 = wrap_for_odejl(compute_rhs_carray_wrapper)
    RHSVersions.compute_rhs_v5(udot, u0, p, 0.0)
    w1(udot, u0, p, 0.0)
    w2(udot, u0, p, 0.0)

    benchmark_this_version("v5", RHSVersions.compute_rhs_v5, u0, tspan, p)
    benchmark_this_version("cwrapper-oif", w1, u0, tspan, p)
    benchmark_this_version("cwrapper-carray", w2, u0, tspan, p)

    # @test udot_v5 ≈ udot_oif_cwrapper rtol=1e-14 atol=1e-14
    # @test udot_v5 ≈ udot_carray_cwrapper rtol=1e-14 atol=1e-14
    # @printf "Leftmost udot_1 value: %.16f\n" udot_v5[1]
    # @printf "Leftmost udot_2 value: %.16f\n" udot_oif_cwrapper[1]
    # @printf "Leftmost udot_3 value: %.16f\n" udot_carray_cwrapper[1]
end

measure()
