using CSV
using DataFrames
using OrdinaryDiffEq
using Plots
using Printf
using Statistics

include("../../helpers.jl")

RESOLUTIONS_LIST = [800, 1600, 3200]
N_RUNS = 30

OUTDIR = Helpers.getOutdir(@__FILE__)
RESULT_FILENAME = OUTDIR * "/runtime_vs_resolution_julia.csv"

function compute_rhs(udot, u, p, t)
    dx, = p

    f = 0.5 * u.^2
    c = maximum(abs.(u))  # Local sound speed.
    f_hat = @views 0.5 * (f[1:end-1] + f[2:end]) .- 0.5 * c * (u[2:end] - u[1:end-1])
    f_plus = @views f_hat[2:length(f_hat)]
    f_minus = @views f_hat[1:length(f_hat) - 1]
    
    udot[2:end-1] = -1.0 / dx * (f_plus - f_minus)

    local_ss_rb = max(abs(u[1]), abs(u[end]))
    f_rb = 0.5 * (f[1] + f[end]) - 0.5 * local_ss_rb * (u[1] - u[end])
    f_lb = f_rb

    udot[1] = -1.0 / dx * (f_minus[1] - f_lb)
    udot[end] = -1.0 / dx * (f_rb - f_plus[end])
end

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

    odeProblem = ODEProblem(compute_rhs, u0, (t0, tfinal), (dx,))

    times = collect(range(t0, tfinal, 101))

    local solution_last_1::Vector{Float64}
    local t_last = -1.0
    solver = init(odeProblem, DP5(); reltol = 1e-6, abstol = 1e-12, save_everystep=false)
    tic = time_ns()
    for t in times[2:end]
        step!(solver, t - solver.t, true)
        solution_last_1 = solver.u
    end
    toc = time_ns()
    runtime_1 = toc - tic
    
    tic = time_ns()
    solution = solve(odeProblem, DP5(), reltol = 1e-6, abstol = 1e-12, saveat=times)
    solution_last_2 = solution.u[end]
    toc = time_ns()
    runtime_2 = toc - tic

    return runtime_1, runtime_2, solution_last_1, solution_last_2
end

function run()
    measure_perf_once(RESOLUTIONS_LIST[1])  # We need to warm up Julia

    table = ["method/resolution", "step!", "solve"]
    solution_last_1 = []
    solution_last_2 = []

    for N in RESOLUTIONS_LIST
        @printf "Resolution %d: estimating runtime from %d runs\n" N N_RUNS
        elapsed_times_1 = []
        elapsed_times_2 = []
        for k = 1:N_RUNS
            runtime_1, runtime_2, solution_last_1, solution_last_2 = measure_perf_once(N)
            push!(elapsed_times_1, runtime_1 / 1e9)
            push!(elapsed_times_2, runtime_2 / 1e9)
        end

        runtime_mean_1, ci_1 = runtime_stats(elapsed_times_1)
        @printf "--- Resolution %d, using step!\n" N
        @printf "Runtime, sec: %.4f ± %.4f\n" runtime_mean_1 ci_1
        @printf "Solution second point from the left value: %.16f\n" solution_last_1[2]

        runtime_mean_2, ci_2 = runtime_stats(elapsed_times_2)
        @printf "--- Resolution %d, using solve\n" N
        @printf "Runtime, sec: %.4f ± %.4f\n" runtime_mean_2 ci_2
        @printf "Solution second point from the left value: %.16f\n" solution_last_2[2]

        val_1 = @sprintf "%.4f ± %.4f" runtime_mean_1 ci_1
        val_2 = @sprintf "%.4f ± %.4f" runtime_mean_2 ci_2
        column = [N, val_1, val_2]
        table = [table column]
    end

    @printf "\n\nSANITY check: two solutions must agree\n"
    @printf "1. Solution second point from the left value: %.16f\n" solution_last_1[2]
    @printf "2. Solution second point from the left value: %.16f\n" solution_last_2[2]

    df = DataFrame(table[2:end, :], Symbol.(table[1, :]))
    CSV.write(RESULT_FILENAME, df)
    println("Performance study results are written to the file ", RESULT_FILENAME)
end

run()
