using Libdl
using OrdinaryDiffEq
using Printf
using Statistics
using Test

N_TRIALS = 30
N::Int = 3200

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

function benchmark_this_version(func, u0, tspan, p, save_solution)
    runtimes::Vector{Float64} = []
    times = collect(range(tspan[1], tspan[2], 101))

    odeProblem = ODEProblem(func, u0, tspan, p)
    solver = init(odeProblem, DP5(); reltol=1e-6, abstol=1e-12, save_everystep = false)
    # Warm up the function under benchmark.
    # func(udot, u0, p, 0.0)
    tic = time_ns()
    for t in times[2:end]
        step!(solver, t - solver.t, true)
    end
    toc = time_ns()
    elapsed = (toc - tic) / 1.0e9

    if save_solution
        solution_filename = @sprintf("_output/N=%04d/solution-julia.txt", N)
        save_vector_to_file(solver.u, solution_filename)
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
    compute_rhs_oif_wrapper = CallbackWrapper.make_wrapper_over_oif_c_callback(compute_rhs_oif_fn)

    @printf "Solving ODEs, statistics from %d trials\n" N_TRIALS

    w1 = wrap_for_odejl(compute_rhs_oif_wrapper)

    save_solution = false
    runtime = benchmark_this_version(w1, u0, tspan, p, save_solution)
    runtimes::Vector{Float64} = []
    for k = 1:N_TRIALS
        if k == N_TRIALS
            save_solution = true
        end
        runtime = benchmark_this_version(w1, u0, tspan, p, save_solution)
        push!(runtimes, runtime)
    end

    mean, ci = runtime_stats(runtimes)
    label = @sprintf "Runtime, sec: "
    print_runtime(label, mean, ci)

    runtime_filename = @sprintf("_output/N=%04d/runtimes-julia.txt", N)
    save_vector_to_file(runtimes, runtime_filename)
end

measure()
