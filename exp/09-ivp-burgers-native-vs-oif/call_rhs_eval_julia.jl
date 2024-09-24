using Printf
using Statistics

N_TRIALS = 30
N_RUNS = 100_000
N = 3200


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


function compute_rhs_v2(udot::Vector{Float64}, u::Vector{Float64}, p::Tuple, t::Float64)
    dx, = p
    dx_inv = inv(dx)
    N = length(u)
    
    c = 0.0  # Local sound speed
    for i = 1:N
        cand = abs(u[i])
        if cand > c
            c = cand
        end
    end
    local_ss_rb = max(abs(u[1]), abs(u[end]))
        
    f_cur = 0.5 * u[1]^2
    f_hat_lb = 0.5 * ((f_cur + 0.5 * u[end]^2) - local_ss_rb * (u[1] - u[end]))
    f_hat_prev = f_hat_lb
    @inbounds for i = 1:N-1
        f_next = 0.5 * u[i+1]^2
        f_hat_cur = 0.5 * ((f_cur+f_next) - c * (u[i+1]-u[i]))
        udot[i] = dx_inv * (f_hat_prev - f_hat_cur)
        f_hat_prev, f_cur = f_hat_cur, f_next
    end
    f_hat_rb = f_hat_lb
    udot[end] = dx_inv * (f_hat_prev - f_hat_rb)
end

function compute_rhs_v3(udot, u, p, t)
    dx, = p
    dx_inv = inv(dx)

    c = maximum(abs, u)  # Local sound speed
    local_ss_rb = max(abs(u[1]), abs(u[end]))

    f_cur = 0.5 * u[1]^2
    f_hat_lb = 0.5 * (f_cur + 0.5 * u[end]^2) - 0.5 * local_ss_rb * (u[1] - u[end])
    f_hat_prev = f_hat_lb
    @inbounds for i = 1:length(udot)-1
        f_next = 0.5 * u[i+1]^2
        f_hat_cur = 0.5 * ((f_cur+f_next) - c * (u[i+1]-u[i]))
        udot[i] = dx_inv * (f_hat_prev - f_hat_cur)
        f_hat_prev, f_cur = f_hat_cur, f_next
    end
    f_hat_rb = f_hat_lb
    udot[end] = dx_inv * (f_hat_prev - f_hat_rb)
end

function compute_rhs_v4(udot::AbstractVector{T}, u::AbstractVector{T}, p::Tuple, t::T) where {T}
    dx = p[1]  # Directly access the first element of the tuple.
    dx_inv = inv(dx)
    N = length(udot)

    c = maximum(abs, u)  # abs, u applies abs without creating a temp array.
    local_ss_rb = max(abs(u[1]), abs(u[end]))

    f_cur = T(0.5) * u[1]^2
    f_hat_lb = T(0.5) * (f_cur + T(0.5) * u[N]^2) - T(0.5) * local_ss_rb * (u[1] - u[N])
    f_hat_prev = f_hat_lb
    @inbounds for i = 1:N-1
        f_next = T(0.5) * u[i+1]^2
        f_hat_cur = T(0.5) * ((f_cur+f_next) - c * (u[i+1]-u[i]))
        udot[i] = dx_inv * (f_hat_prev - f_hat_cur)
        f_hat_prev, f_cur = f_hat_cur, f_next
    end
    udot[N] = dx_inv * (f_hat_prev - f_hat_lb)
end

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

    @printf "Julia, accumulated runtime of %d RHS evals, statistics from %d trials\n" N_RUNS N_TRIALS
    @printf "Problem size is %d\n" length(udot)

    benchmark_this_version("v1", compute_rhs_v1, udot_v1, u, p)
    benchmark_this_version("v2", compute_rhs_v2, udot_v2, u, p)
    benchmark_this_version("v3", compute_rhs_v3, udot_v3, u, p)
    benchmark_this_version("v4", compute_rhs_v4, udot_v4, u, p)

    @test udot_v1 ≈ udot_v2 rtol=1e-14 atol=1e-14
    @test udot_v1 ≈ udot_v3 rtol=1e-14 atol=1e-14
    @test udot_v1 ≈ udot_v4 rtol=1e-14 atol=1e-14
end

measure()
