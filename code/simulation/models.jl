module Models


using Graphs
using Graphs.SimpleGraphs
using Distributions
using LinearAlgebra
using Plots


export CNVM
export simulateSSA, simulateEnsembleSSA
export computeCvFromTrajectory, computeMeanVariance
export unifyEnsembleTime, plotEnsemble


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


function simulateEnsembleSSA(params::CNVM, t_max, x_init, n)

    x_traj_list = Array{Vector}(undef, n)
    t_traj_list = Array{Vector}(undef, n)

    Threads.@threads for i = 1:n
        x_traj_list[i], t_traj_list[i] = simulateSSA(params, t_max, copy(x_init))
    end
    return x_traj_list, t_traj_list
end


function simulateSSA(x_init::Vector,
                  t_max,
                  num_states,
                  neighbor_list,
                  r_imit,
                  r_noise,
                  prob_imit,
                  prob_noise,
                  degrees  
                    )

    num_nodes = length(degrees)

    # Setting up exponential RV X with E[X] = event rate
    next_event_rate = 1 / (r_imit * sum(degrees)+ r_noise * num_nodes)
    event_distance_distribution = Exponential(next_event_rate)

    noise_probability = r_noise * num_nodes * next_event_rate
    
    prob_table, alias_table = buildAliasTable(degrees)

    t=0
    x_traj = [copy(x_init)]
    t_traj = [0.]

    x = x_init

    while t < t_max
        t += rand(event_distance_distribution)
        noise = rand() < noise_probability ? true : false 

        if noise
            node = rand(1:num_nodes)
            new_state = rand(1:num_states)
            if rand() < prob_noise[x[node], new_state]
                x[node] = new_state
            end
        else
            node = sampleFromAlias(prob_table, alias_table)
            neighbors = neighbor_list[node]
            rand_neighbor = neighbors[rand(1:length(neighbors))]
            new_state = x[rand_neighbor]
            if rand() < prob_imit[x[node], new_state]
                x[node] = new_state
            end
        end

        push!(x_traj, copy(x))
        push!(t_traj, t)

    end

    return x_traj, t_traj
end


function simulateSSA(params::CNVM, t_max, x_init)

    neighbor_list = [all_neighbors(params.network, i) for i in vertices(params.network) ]
    
    simulateSSA(x_init,
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


function buildAliasTable(weights)  
    # Algorithm from Kronmal and Peterson; see alias method for more information 
    table_prob = weights / sum(weights) * length(weights)
    table_alias = ones(Integer, length(weights))

    small_ids = [i for i=1:length(table_prob) if table_prob[i] < 1]
    large_ids = [i for i=1:length(table_prob) if table_prob[i] >= 1]

    while length(small_ids) > 0 && length(large_ids) > 0
        small_id = pop!(small_ids)
        large_id = pop!(large_ids)

        table_alias[small_id] = large_id
        table_prob[large_id] = table_prob[large_id] + table_prob[small_id] -1

        if table_prob[large_id] < 1
            push!(small_ids, large_id)
        else
            push!(large_ids, large_id)
        end
    end

    table_prob[large_ids] .= 1
    table_prob[small_ids] .= 1

    return table_prob, table_alias
end


function sampleFromAlias(table_prob::Vector{Float64}, table_alias::Vector{Integer})

    x = rand()
    idx = trunc(Int, x * length(table_prob))
    y = length(table_prob) * x - idx
    if y < table_prob[idx+1]
        return idx+1
    end

    return table_alias[idx+1]

end


function computeCvFromTrajectory(trajectory::Vector{Vector{Int64}}, num_states)

    cv = zeros(Int64,(size(trajectory)[1], num_states))
    for i in 1:size(trajectory)[1]
        for state in trajectory[i]
         cv[i, state] += 1
        end 
    end 
    return cv
end 


function computeCvFromTrajectory(trajectory::Vector{Vector}, num_states)
    """Function for Vector of trajectories"""
    
    n = length(trajectory)

    cvs = Array{AbstractMatrix}(undef, n)

    for i = 1:n
        cvs[i] = computeCvFromTrajectory(trajectory[i], num_states)
    end

    return cvs
end


function plotEnsemble(ensemble_cv, ensemble_time, n_points, t_max)
    
    n_traj = size(ensemble_cv)[1]
    n_points = maximum([size(ensemble_cv[1])[1]])
    n_states = size(ensemble_cv[1])[2]
    
    t_traj, cv_traj = unifyEnsembleTime(ensemble_cv, ensemble_time, n_points, t_max)

    mean, variance = computeMeanVariance(cv_traj)

    for j = 1:n_states
        plt = plot(title="State $j", )
        for i = 1:n_traj
            plot!( t_traj, cv_traj[:,j,i], color="light gray")
        end
        plot!(t_traj, mean[:,j], color="red", legend=false)
        plot!(t_traj, mean[:,j] + variance[:,j], color="orange")
        plot!(t_traj, mean[:,j] - variance[:,j], color="orange")
        display(plt)    
    
    end

    
end

"""
Unifies the time series of several individual trajectories.
The trajectories will be sampled at the new time points.
The number of points of the unified series is equal to the maximum of time points over all passed series.
The unified points will have equidistant spacing.
At every new time point every trajectory will be sampled. 
For this the first value to the left is used. 
In case of trajectories computed by the Gillespie algorithm this does not change the statistical correctness. 
"""
function unifyEnsembleTime(ensemble_cv, ensemble_time, n_points, t_max )

    n_traj = length(ensemble_cv)
    n_states = size(ensemble_cv[1])[2]
    
    time_series = [i/n_points*t_max for i=0:n_points-1]

    new_traj = zeros(UInt64, (n_points, n_states, n_traj))    

    for i = 1:n_traj
        for j = 1:n_states

            # index for the input trajectory
            old_index = 1 
            # index for the new unified trajectory
            new_index = 1

            n_old_time = size(ensemble_cv[i])[1]

            while new_index < n_points && old_index < n_old_time

                if ensemble_time[i][old_index] >= time_series[new_index]
                    new_traj[new_index, j, i] = ensemble_cv[i][old_index, j]
                    new_index += 1
                else
                    old_index += 1
                end
            end 

            while new_index <= n_points
                new_traj[new_index,j,i] = ensemble_cv[i][old_index,j]
                new_index += 1
            end 
    
        end
    end
    return time_series, new_traj
        
end


function computeMeanVariance(cv_traj)

    n_traj = size(cv_traj)[3]
    n_states = size(cv_traj)[2]
    n_points = size(cv_traj)[1]
    mean = zeros(Float64,(n_points, n_states))
    variance = zeros(Float64,(n_points, n_states))

    for i = 1:n_traj
        for j = 1:n_states
        mean[:,j] .+= cv_traj[:,j,i]
        end
    end
    mean ./= n_traj

    for i = 1:n_traj
        for j = 1:n_states
            variance[:,j] .+= (mean[:,j] .- cv_traj[:,j,i]).^2

        end
    end

    variance ./= n_traj
    variance = sqrt.(variance)

    return mean, variance
end 

end