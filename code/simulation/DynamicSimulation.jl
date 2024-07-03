
module DynamicsOnNetworkSimulation

include("Gillespie.jl")

using Graphs
using Graphs.SimpleGraphs
using LinearAlgebra
using Plots
using Distributions



export CNVM
export gillespie, ensembleGillespie
export computeCvFromTrajectory, computeCvFromTrajectories, unifyEnsembleTime
export eulerMaruyama, ensembleEulerMaruyama


struct CNVM
    num_states::Integer
    r_imit::Real
    r_noise::Real
    prob_imit::Matrix
    prob_noise::Matrix
    network::SimpleGraph
end


function CNVM(num_states::Integer, r::Matrix, rt::Matrix, network::SimpleGraph)
    # Change parameters to a form which is better suited for computation

    new_r = copy(r)
    new_r[diagind(new_r)] .= 0

    # rate at which imitations will occur according to principle of competing exponentials
    event_rate_imit = maximum(new_r)

    if event_rate_imit > 0
        # normalize rates by overall event rate to receive probabilities_imitation (prob_imit)
        prob_imit = new_r / event_rate_imit
    else
        prob_imit = zeros((num_states, num_states))
    end

    new_rt = copy(rt)
    new_rt[diagind(new_rt)] .= 0

    event_rate_noise = maximum(new_rt) * num_states # analog to imitation above

    if event_rate_noise > 0
        prob_noise = new_rt * num_states / event_rate_noise
    else
        prob_noise = zeros((num_states, num_states))
    end

    CNVM(num_states, event_rate_imit, event_rate_noise, prob_imit, prob_noise, network)
end


function ensembleGillespie(params::CNVM, t_max, x_init, n_runs)

    x_traj_list = Array{Vector}(undef, n_runs)
    t_traj_list = Array{Vector}(undef, n_runs)

    Threads.@threads for i = 1:n_runs
        t_traj_list[i], x_traj_list[i] = gillespie(params, t_max, copy(x_init))
        
    end
    return t_traj_list, x_traj_list
end


function gillespie(params::CNVM, t_max, x_init)

    neighbor_list = [all_neighbors(params.network, i) for i in vertices(params.network)]

    gillespie(x_init,
        t_max,
        params.num_states,
        neighbor_list,
        params.r_imit,
        params.r_noise,
        params.prob_imit,
        params.prob_noise,
        degree(params.network)
    )
end


function computeCvFromTrajectory(trajectory, num_states)

    cv = zeros(Int64, (size(trajectory)[1], num_states))

    for i in 1:size(trajectory)[1]
        for state in trajectory[i]
            cv[i, state] += 1
        end
    end
    return cv
end


function computeCvFromTrajectories(trajectory, num_states)

    n = length(trajectory)

    cvs = Array{AbstractMatrix}(undef, n)

    for i = 1:n
        cvs[i] = computeCvFromTrajectory(trajectory[i], num_states)
    end

    return cvs
end


"""
Unifies the time series of several individual trajectories.
Assumes every time series starts at 0.
The trajectories will be sampled at the new time points.
The number of points of the unified series is equal to the maximum of time points over all passed series.
The unified points will have equidistant spacing.
At every new time point every trajectory will be sampled. 
For this the first value to the left is used. 
"""
function unifyEnsembleTime(
    ensemble_time,
    ensemble_traj,
    new_timesteps::Vector
)

    n_traj = length(ensemble_traj)
    n_states = size(ensemble_traj[1])[2]

    
    n_new_time = length(new_timesteps)

    new_traj = zeros(UInt64, (n_new_time, n_states, n_traj))

    for i = 1:n_traj
        for j = 1:n_states

            # index for the input trajectory
            old_index = 1
            # index for the new unified trajectory
            new_index = 1

            n_old_time = size(ensemble_traj[i])[1]

            while new_index < n_new_time && old_index < n_old_time - 1

                if new_timesteps[new_index] <= ensemble_time[i][old_index+1]
                    new_traj[new_index, j, i] = ensemble_traj[i][old_index, j]
                    new_index += 1
                else
                    old_index += 1
                end

            end

            while new_index <= n_new_time
                new_traj[new_index, j, i] = ensemble_traj[i][n_old_time, j]
                new_index += 1
            end

        end
    end

    return new_timesteps, new_traj
end


function unifyEnsembleTime(
    ensemble_time,
    ensemble_traj,
    t_max,
    n_new_timesteps
)
    new_timesteps = [i / n_new_timesteps * t_max for i = 0:n_new_timesteps-1]

    unifyEnsembleTime(ensemble_time, ensemble_traj, new_timesteps)

end


function brownianMotion(timesteps::Vector, d=1)

    n = length(timesteps)
    normal_dist = Normal()

    if length(d) > 0
        dim = Tuple(collect(Iterators.flatten((n, d))))
    else
        dim = (n, d)
    end


    rand_incs = rand(normal_dist, dim)
    motion = zeros(dim)
    for i = 2:n
        motion[i, :, :] = motion[i-1, :, :] + sqrt(timesteps[i] - timesteps[i-1]) .* rand_incs[i-1, :, :]

    end
    return motion
end


function eulerMaruyama(drift, diffusion, dim, rand_dim, x0, timesteps::Vector)

    n_steps = length(timesteps)
    traj = zeros(n_steps, dim)
    traj[1, :] .= copy(x0)
    randinc = brownianMotion(timesteps, rand_dim)

    for i = 1:n_steps-1

        Δt = timesteps[i+1] - timesteps[i]

        driftterm = drift(traj[i, :]) * Δt

        diffuterm = diffusion(traj[i, :]) * (randinc[i+1, :, :] - randinc[i, :, :])

        traj[i+1, :] = traj[i, :] + driftterm + diffuterm

    end
    return timesteps, traj
end


function eulerMaruyama(drift, diffusion, dim, rand_dim, x0, t0, tmax, n_timesteps)

    timesteps = [t0 + i * (tmax - t0) / n_timesteps for i = 0:n_timesteps-1]

    return timesteps, eulerMaruyama(drift, diffusion, dim, rand_dim, x0, time)

end


function ensembleEulerMaruyama(drift, diffusion, n_states, rand_dim, x0, t0, tmax, n_timesteps, n_runs)

    timesteps = [t0 + i * (tmax - t0) / n_timesteps for i = 0:n_timesteps-1]

    x_traj_list = zeros(length(timesteps), n_states, n_runs)

    Threads.@threads for i = 1:n_runs
        timesteps, x_traj_list[:, :, i] = eulerMaruyama(drift, diffusion, n_states, rand_dim, x0, timesteps)
    end
    return timesteps, x_traj_list
end

end