using CSV
using DataFrames
using OrdinaryDiffEq
using Plots
using Printf
using Statistics
using Test

include("../../helpers.jl")
include("rhsversions.jl")
using .RHSVersions

VERSIONS = ["v1", "v2", "v3", "v4", "v5"]
RESOLUTIONS_LIST = [200, 400, 800, 1600, 3200]
N_RUNS = 30

OUTDIR = Helpers.getOutdir(@__FILE__)
RESULT_FILENAME_JULIA = OUTDIR * "/runtime_vs_resolution_julia.csv"

function runtime_stats(elapsed_times)
    runtime_mean = mean(elapsed_times)
    runtime_std = std(elapsed_times; corrected=true, mean=runtime_mean)
    sem = runtime_std / sqrt(length(elapsed_times))
    ci = 2 * sem

    return runtime_mean, ci
end

function measure_perf_once(N)
    x = collect(range(0, 2, N + 1))
    dx = 2 / N
    u0 = 0.5 .- 0.25 * sin.(pi * x)

    CFL = 0.5
    dt_max = dx * CFL

    t0 = 0.0
    tfinal = 10.0

    times = collect(range(t0, tfinal, 101))

    runtimes = Dict()
    solutions = Dict{String, Float64}()

    local solution_last_1::Vector{Float64}
    for v in VERSIONS
        if v == "v1"
            odeProblem = ODEProblem(compute_rhs_v1, u0, (t0, tfinal), (dx,))
        elseif v == "v2"
            odeProblem = ODEProblem(compute_rhs_v2, u0, (t0, tfinal), (dx,))
        elseif v == "v3"
            odeProblem = ODEProblem(compute_rhs_v3, u0, (t0, tfinal), (dx,))
        elseif v == "v4"
            odeProblem = ODEProblem(compute_rhs_v4, u0, (t0, tfinal), (dx,))
        elseif v == "v5"
            odeProblem = ODEProblem(compute_rhs_v5, u0, (t0, tfinal), (dx,))
        end
        solver = init(odeProblem, DP5(); reltol = 1e-6, abstol = 1e-12, save_everystep=false)
        tic = time_ns()
        for t in times[2:end]
            step!(solver, t - solver.t, true)
            solution_last_1 = solver.u
        end
        toc = time_ns()
        @printf "RHS %s: RHS evals = %d\n" v solver.stats.nf
        @printf "RHS %s: accepted  = %d\n" v solver.stats.naccept
        @printf "RHS %s: rejected  = %d\n" v solver.stats.nreject
        runtimes[v] = toc - tic
        @printf "%s leftmost point = %.16f\n" v solver.u[1]
        solutions[v] = solver.u[1]
    end

    return runtimes, solutions
end

function main()
    dx = 0.0333
    u = rand(10000)
    result_1 = similar(u)
    result_2 = similar(u)
    result_3 = similar(u)
    result_4 = similar(u)
    result_5 = similar(u)
    compute_rhs_v1(result_1, u, (dx,), 0.0)
    compute_rhs_v2(result_2, u, (dx,), 0.0)
    compute_rhs_v3(result_3, u, (dx,), 0.0)
    compute_rhs_v4(result_4, u, (dx,), 0.0)
    compute_rhs_v5(result_5, u, (dx,), 0.0)
    @test result_1 ≈ result_2 rtol=1e-14 atol=1e-14
    @test result_1 ≈ result_3 rtol=1e-14 atol=1e-14
    @test result_1 ≈ result_4 rtol=1e-14 atol=1e-14
    @test result_1 ≈ result_5 rtol=1e-14 atol=1e-14

    measure_perf_once(RESOLUTIONS_LIST[1])  # We need to warm up Julia

    label_1 = @sprintf "%-30s" "jl-native-v1"
    label_2 = @sprintf "%-30s" "jl-native-v2"
    label_3 = @sprintf "%-30s" "jl-native-v3"
    label_4 = @sprintf "%-30s" "jl-native-v4"
    label_5 = @sprintf "%-30s" "jl-native-v5"

    table = ["method/resolution", label_1, label_2, label_3, label_4, label_5]

    solution_last_1 = []
    solution_last_2 = []

    for N in RESOLUTIONS_LIST
        @printf "Resolution %d: estimating runtime from %d runs\n" N N_RUNS
        elapsed_times = Dict()
        local solutions::Dict{String, Float64}
        for v in VERSIONS
            elapsed_times[v] = []
        end

        for k = 1:N_RUNS
            runtimes, solutions = measure_perf_once(N)
            for v in VERSIONS
                push!(elapsed_times[v], runtimes[v] / 1e9)
            end
        end

        @printf "--- Resolution %d\n" N
        for v in VERSIONS
            @printf "%-16s: leftmost point of solution = %.16f\n" v solutions[v]
        end

        column = [string(N)]
        for v in VERSIONS
            runtime_mean, ci = runtime_stats(elapsed_times[v])
            @printf "%-16s: runtime, sec = %.2f ± %.2f\n" v runtime_mean ci

            val = @sprintf "%.2f ± %.2f" runtime_mean ci
            push!(column, val)
        end
        table = [table column]
    end

    df = DataFrame(table[2:end, :], Symbol.(table[1, :]))
    CSV.write(RESULT_FILENAME_JULIA, df)
    println("Performance study results are written to the file ", RESULT_FILENAME_JULIA)
end

main()
