using CSV
using DataFrames
using OrdinaryDiffEq
using Plots
using Printf
using Statistics
using Test

include("../../helpers.jl")

VERSIONS = ["v1", "v2", "v3"]
RESOLUTIONS_LIST = [800, 1600, 3200]
RESOLUTIONS_LIST = [800, 1600]
N_RUNS = 2

OUTDIR = Helpers.getOutdir(@__FILE__)
RESULT_FILENAME_JULIA = OUTDIR * "/runtime_vs_resolution_julia.csv"

function compute_rhs_v1(udot, u, p, t)
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

function compute_rhs_v2(udot, u, p, t)
    dx, = p
    N = length(u)

    f = similar(u)
    for i = 1:N
        f[i] = 0.5 * u[i]^2
    end

    # Local sound speed.
    c = 0.0
    for i = 1:N
        if abs(u[i]) > c
            c = abs(u[i])
        end
    end

    f_hat = Array{Float64}(undef, N - 1)
    for i = 1:N-1
        f_hat[i] = 0.5 * (f[i] + f[i + 1]) - 0.5 * c * (u[i + 1] - u[i])
    end

    for i = 2:N-1
        udot[i] = -1.0 / dx * (f_hat[i] - f_hat[i - 1])
    end

    local_ss_rb = max(abs(u[1]), abs(u[end]))
    f_rb = 0.5 * (f[1] + f[end]) - 0.5 * local_ss_rb * (u[1] - u[end])
    f_lb = f_rb

    udot[1] = -1.0 / dx * (f_hat[1] - f_lb)
    udot[end] = -1.0 / dx * (f_rb - f_hat[end])
end

function compute_rhs_v3(udot, u, p, t)
    dx, = p
    dx⁻¹ = inv(dx)
    
    c = maximum(abs, u)  # Local sound speed
    local_ss_rb = max(abs(u[1]), abs(u[end]))
        
    f_cur = 0.5 * u[1]^2
    f̂_lb = 0.5 * (f_cur + 0.5 * u[end]^2) - 0.5 * local_ss_rb * (u[1] - u[end])
    f̂_prev = f̂_lb
    @inbounds for i = 1:length(udot)-1
        f_next = 0.5 * u[i+1]^2
        f̂_cur = 0.5 * ((f_cur+f_next) - c * (u[i+1]-u[i]))
        udot[i] = dx⁻¹ * (f̂_prev - f̂_cur)
        f̂_prev, f_cur = f̂_cur, f_next
    end
    f̂_rb = f̂_lb
    udot[end] = dx⁻¹ * (f̂_prev - f̂_rb)
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

    times = collect(range(t0, tfinal, 101))

    runtimes = Dict()

    local solution_last_1::Vector{Float64}
    for v in VERSIONS
        if v == "v1"
            odeProblem = ODEProblem(compute_rhs_v1, u0, (t0, tfinal), (dx,))
        elseif v == "v2"
            odeProblem = ODEProblem(compute_rhs_v2, u0, (t0, tfinal), (dx,))
        elseif v == "v3"
            odeProblem = ODEProblem(compute_rhs_v3, u0, (t0, tfinal), (dx,))
        end
        solver = init(odeProblem, DP5(); reltol = 1e-6, abstol = 1e-12, save_everystep=false)
        tic = time_ns()
        for t in times[2:end]
            step!(solver, t - solver.t, true)
            solution_last_1 = solver.u
        end
        toc = time_ns()
        runtimes[v] = toc - tic
    end

    return runtimes, solution_last_1
end

function main()
    dx = 0.0333
    u = rand(10000)
    result_1 = similar(u)
    result_2 = similar(u)
    result_3 = similar(u)
    compute_rhs_v1(result_1, u, (dx,), 0.0)
    compute_rhs_v2(result_2, u, (dx,), 0.0)
    compute_rhs_v3(result_3, u, (dx,), 0.0)
    @test result_1 ≈ result_2 rtol=1e-14 atol=1e-14
    @test result_1 ≈ result_3 rtol=1e-14 atol=1e-14

    measure_perf_once(RESOLUTIONS_LIST[1])  # We need to warm up Julia

    table = ["method/resolution", "jl-plain", "jl-loops", "jl-fused-loops"]

    solution_last_1 = []
    solution_last_2 = []

    for N in RESOLUTIONS_LIST
        @printf "Resolution %d: estimating runtime from %d runs\n" N N_RUNS
        elapsed_times = Dict()
        for v in VERSIONS
            elapsed_times[v] = []
        end

        for k = 1:N_RUNS
            runtimes, solution_last_1 = measure_perf_once(N)
            for v in VERSIONS
                push!(elapsed_times[v], runtimes[v] / 1e9)
            end
        end

        column = [string(N)]
        for v in VERSIONS
            runtime_mean, ci = runtime_stats(elapsed_times[v])
            @printf "--- Resolution %d, version %s\n" N v
            @printf "    Runtime, sec: %.3f ± %.3f\n" runtime_mean ci
            @printf "Solution second point from the left value: %.16f\n" solution_last_1[1]

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
