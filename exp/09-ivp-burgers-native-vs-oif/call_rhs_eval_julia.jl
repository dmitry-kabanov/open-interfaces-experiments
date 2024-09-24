using Printf
using Statistics
using Test

N_TRIALS = 30
N_RUNS = 100_000
N = 3200

include("rhsversions.jl")

function benchmark_this_version(version_name, func, udot, u, p)
    runtimes = []
    dx = p[1]

    # Warm up the function under benchmark.
    func(udot, u[:, 1], p, 0.0)

    for t = 1:N_TRIALS
        tic = time_ns()
        for j = 1:N_RUNS
            func(udot, u[:, j], (dx,), 0.0)
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

    udot = similar(u0)
    udot_v1 = similar(u0)
    udot_v2 = similar(u0)
    udot_v3 = similar(u0)
    udot_v4 = similar(u0)
    u = rand(N + 1, N_RUNS)
    for j = 1:N_RUNS
        u[:, j] = u0
    end

    @printf "Julia, accumulated runtime of %d RHS evals, statistics from %d trials\n" N_RUNS N_TRIALS
    @printf "Problem size is %d\n" length(udot)

    benchmark_this_version("v1", compute_rhs_v1, udot_v1, u, p)
    benchmark_this_version("v2", compute_rhs_v2, udot_v2, u, p)
    benchmark_this_version("v3", compute_rhs_v3, udot_v3, u, p)
    benchmark_this_version("v4", compute_rhs_v4, udot_v4, u, p)

    @test udot_v1 ≈ udot_v2 rtol=1e-14 atol=1e-14
    @test udot_v1 ≈ udot_v3 rtol=1e-14 atol=1e-14
    @test udot_v1 ≈ udot_v4 rtol=1e-14 atol=1e-14
    @printf "Leftmost udot value: %.16f" udot_v4[1]
end

measure()
