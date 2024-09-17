using Printf
using Statistics

N_TRIALS = 30
N_RUNS = 100_000


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

function measure()
    N = 4000
    x = collect(range(0, 2, N + 1))
    dx = 2 / N
    u0 = 0.5 .- 0.25 * sin.(pi * x)

    CFL = 0.5
    dt_max = dx * CFL

    t0 = 0.0
    tfinal = 10.0

    udot = similar(u0)
    u = rand(N + 1, N_RUNS)

    @printf "Julia, accumulated %d RHS evaluations, %d trials\n" N_RUNS N_TRIALS

    # Timing the plain version
    values_plain = []
    compute_rhs_v1(udot, u[:, 1], (dx,), 0.0)
    for t = 1:N_TRIALS
        tic = time_ns()
        for j = 1:N_RUNS
            compute_rhs_v1(udot, u[:, j], (dx,), 0.0)
        end
        toc = time_ns()
        elapsed = (toc - tic) / 1e9
        push!(values_plain, elapsed)
    end
    mean, ci = runtime_stats(values_plain)
    @printf "Julia, plain version: %.3f ± %.3f\n" mean ci

    # Timing the optimized version
    values_optim = []
    compute_rhs_v2(udot, u[:, 1], (dx,), 0.0)
    for t = 1:N_TRIALS
        tic = time_ns()
        for j = 1:N_RUNS
            compute_rhs_v2(udot, u[:, j], (dx,), 0.0)
        end
        toc = time_ns()
        elapsed = (toc - tic) / 1e9
        push!(values_optim, elapsed)
    end
    mean, ci = runtime_stats(values_optim)
    @printf "Julia, optim version: %.3f ± %.3f\n" mean ci
end

measure()
