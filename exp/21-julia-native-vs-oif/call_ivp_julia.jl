# This script benchmarks the Julia implementation of the Open Interfaces user code.
#
# Precisely, we compare the runtime of solving the Burgers' equation
# directly and via Open Interfaces.

using Libdl
using OrdinaryDiffEq
using Printf
using Statistics
using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.IVP

include("rhsversions.jl")
using .RHSVersions
include("callback.jl")
using .CallbackWrapper

function oif_wrapper(func)
    function wrapper(t, u, udot, p)
        func(udot, u, p, t)
    end

    return wrapper
end

function compute_initial_condition(N::Int)
    a = 0.0
    b = 2.0
    dx = (b - a) / N
    x = Vector{Float64}(undef, N + 1)

    # The grid has the following structure in C:
    # 0 -- 1  -- 2 -- .. -- N - 1 -- N
    # which gives in total N + 1 points.
    for i = 1:N+1
        x[i] = a + (i - 1) * dx;
    end
    u0 = 0.5 .- 0.25 * sin.(pi * x)

    return u0, x, dx
end

function benchmark_native_version(func, N::Int, save_ic::Bool, save_solution::Bool)
    u0, x, dx = compute_initial_condition(N)

    if save_ic
        ic_filename = @sprintf("_output/N=%04d/ic-julia-native.txt", N)
        save_solution_to_file(x, u0, ic_filename)
    end

    t0 = 0.0
    tfinal = 10.0
    p = (dx, )
    tspan = (t0, tfinal)
    times = collect(range(t0, tfinal, 101))

    odeProblem = ODEProblem(func, u0, tspan, p)
    solver = init(odeProblem, DP5(); reltol = 1e-6, abstol = 1e-12, save_everystep = false)
    # Warm up the function under benchmark.
    # func(udot, u0, p, 0.0)
    tic = time_ns()
    for t in times[2:end]
        step!(solver, t - solver.t, true)
    end
    toc = time_ns()
    elapsed = (toc - tic) / 1.0e9

    if save_solution
        solution_filename = @sprintf("_output/N=%04d/solution-julia-native.txt", N)
        save_solution_to_file(x, solver.u, solution_filename)
    end

    return elapsed
end

function benchmark_oif_version(func, N::Int, save_ic::Bool, save_solution::Bool)
    u0, x, dx = compute_initial_condition(N)

    if save_ic
        ic_filename = @sprintf("_output/N=%04d/ic-julia-oif.txt", N)
        save_solution_to_file(x, u0, ic_filename)
    end

    t0 = 0.0
    tfinal = 10.0
    p = (dx, )
    tspan = (t0, tfinal)
    times = collect(range(t0, tfinal, 101))

    oif_rhs = oif_wrapper(func)

    s = IVP.Self("jl_diffeq")
    IVP.set_initial_value(s, u0, t0)
    IVP.set_user_data(s, p)
    IVP.set_rhs_fn(s, oif_rhs)
    IVP.set_tolerances(s, 1e-6, 1e-12)

    # Warm up the function under benchmark.
    # func(udot, u0, p, 0.0)
    tic = time_ns()
    for t in times[2:end]
        IVP.integrate(s, t)
    end
    toc = time_ns()
    elapsed = (toc - tic) / 1.0e9

    if save_solution
        solution_filename = @sprintf("_output/N=%04d/solution-julia-oif.txt", N)
        save_solution_to_file(x, s.y, solution_filename)
    end

    return elapsed
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

function save_vector_to_file(vec::Vector{Float64}, filename::String)
    open(filename, "w") do file
        for value in vec
            write(file, "$(value)\n")
        end
    end
end

function save_solution_to_file(grid::Vector{Float64}, sol::Vector{Float64}, filename::String)
    open(filename, "w") do file
        for i = 1:length(grid)
            x = grid[i]
            y = sol[i]
            row = @sprintf "%.16f %.16f\n" x y
            write(file, row)
        end
    end
end

function measure(N::Int, N_TRIALS::Int)
    @printf "N = %04d\n" N
    @printf "N_TRIALS = %d\n" N_TRIALS

    Base.Filesystem.mkpath(@sprintf("_output/N=%04d", N))

    rhs = compute_rhs_v5

    save_ic = false
    save_solution = false

    # Warm up the function under benchmark.
    runtime = benchmark_native_version(rhs, N, save_ic, save_solution)

    runtimes::Vector{Float64} = []
    for k = 1:N_TRIALS
        if k == N_TRIALS
            save_ic = true
            save_solution = true
        end
        runtime = benchmark_native_version(rhs, N, save_ic, save_solution)
        push!(runtimes, runtime)
    end

    mean, ci = runtime_stats(runtimes)
    label = @sprintf "Runtime, sec: "
    print_runtime(label, mean, ci)

    runtime_filename = @sprintf("_output/N=%04d/runtimes-julia-oif.txt", N)
    save_vector_to_file(runtimes, runtime_filename)

    ### Benchmark OIF version.
    save_ic = false
    save_solution = false

    # Warm up the function under benchmark.
    runtime = benchmark_oif_version(rhs, N, save_ic, save_solution)

    runtimes = []
    for k = 1:N_TRIALS
        if k == N_TRIALS
            save_ic = true
            save_solution = true
        end
        runtime = benchmark_oif_version(rhs, N, save_ic, save_solution)
        push!(runtimes, runtime)
    end

    mean, ci = runtime_stats(runtimes)
    label = @sprintf "Runtime, sec: "
    print_runtime(label, mean, ci)

    runtime_filename = @sprintf("_output/N=%04d/runtimes-julia-oif.txt", N)
    save_vector_to_file(runtimes, runtime_filename)
end

N = parse(Int, ARGS[1])
N_TRIALS = parse(Int, ARGS[2])
# N = 320
# N_TRIALS = 1

print("Parsed resolution N = $N, number of trials N_TRIALS = $N_TRIALS\n")

measure(N, N_TRIALS)
